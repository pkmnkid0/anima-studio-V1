"""LoRA tools: slim a LoRA down, and strip the metadata baked into it.

A LoRA layer stores two thin matrices, `down` (r x in) and `up` (out x r). Their
product, scaled by alpha/r, is the change the LoRA makes to the model. Most
LoRAs are trained with more rank than that change really needs, so the extra
rank is mostly noise and wasted file size.

For every layer we find the best possible smaller version of the change
(Eckart-Young: truncate its singular values) and store that. It is done on the
two thin matrices (QR, then an SVD of an r x r matrix), so it is fast and never
builds the full out x in matrix.

Only numpy is needed: safetensors is read and written here directly, which also
handles bfloat16 files that plain numpy cannot load.

Layers that are not plain LoRA (LoHa, LoKr, DoRA, ...) are copied unchanged.

The same reader/writer also powers `metadata()` / `strip_metadata()`: a trained
LoRA carries a `__metadata__` block naming the base model, the dataset and the
training settings, and people who share a file often want that gone without
touching a single weight. Weights are copied byte for byte, and the copy is
re-read and hash-checked before it is reported as done.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAX_BYTES = 3 * 1024 ** 3        # refuse absurd inputs

_DOWN = (".lora_down.weight", ".lora_A.weight")
_UP = (".lora_up.weight", ".lora_B.weight")
_FLOATS = {"F32", "F16", "BF16"}
_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}


class SlimError(RuntimeError):
    pass


# ------------------------------------------------------------ safetensors I/O

def _read(path: str) -> tuple[dict, dict]:
    p = Path(path)
    if not p.is_file():
        raise SlimError(f"File not found: {path}")
    if p.suffix.lower() != ".safetensors":
        raise SlimError("Pick a .safetensors LoRA file.")
    if p.stat().st_size > MAX_BYTES:
        raise SlimError("That file is larger than 3 GB - it does not look like a LoRA.")
    with open(p, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise SlimError("That file is not a valid .safetensors file.")
        (n,) = struct.unpack("<Q", head)
        if n > 100 * 1024 ** 2:
            raise SlimError("That file is not a valid .safetensors file.")
        header_bytes = f.read(n)
        if len(header_bytes) != n:
            raise SlimError("That file is not a valid .safetensors file.")
        try:
            header = json.loads(header_bytes)
        except json.JSONDecodeError as exc:
            raise SlimError("That file is not a valid .safetensors file.") from exc
        if not isinstance(header, dict):
            raise SlimError("That file is not a valid .safetensors file.")
        base = 8 + n
        meta = header.pop("__metadata__", {}) or {}
        out = {}
        for name, info in header.items():
            try:
                a, b = info["data_offsets"]
                dtype, shape = info["dtype"], list(info["shape"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SlimError("That file is not a valid .safetensors file.") from exc
            if not isinstance(a, int) or not isinstance(b, int) or b < a:
                raise SlimError("That file is not a valid .safetensors file.")
            f.seek(base + a)
            out[name] = {"dtype": dtype, "shape": shape, "raw": f.read(b - a)}
    return out, meta


def _to_f32(t: dict) -> np.ndarray:
    d, raw = t["dtype"], t["raw"]
    if d == "F32":
        a = np.frombuffer(raw, dtype="<f4")
    elif d == "F16":
        a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif d == "BF16":
        a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif d == "F64":
        a = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    else:
        raise SlimError(f"Unsupported tensor type {d}.")
    return a.reshape(t["shape"]).astype(np.float32, copy=False)


def _from_f32(a: np.ndarray, dtype: str) -> bytes:
    a = np.ascontiguousarray(a, dtype=np.float32)
    if dtype == "F32":
        return a.astype("<f4").tobytes()
    if dtype == "F16":
        return a.astype("<f2").tobytes()
    if dtype == "BF16":          # round to nearest even
        u = a.view(np.uint32)
        u = (u + (((u >> 16) & 1) + 0x7FFF)) >> 16
        return u.astype("<u2").tobytes()
    raise SlimError(f"Unsupported output type {dtype}.")


def _write(path: str, tensors: dict, meta: dict) -> None:
    header: dict = {}
    if meta:
        header["__metadata__"] = {str(k): str(v) for k, v in meta.items()}
    off = 0
    for name, t in tensors.items():
        n = len(t["raw"])
        header[name] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for t in tensors.values():
            f.write(t["raw"])


# ---------------------------------------------------------------- LoRA layout

def _modules(tensors: dict) -> tuple[list[dict], set[str]]:
    """Find plain LoRA layers. Returns (modules, names of tensors they own)."""
    mods, used = [], set()
    for key in tensors:
        for dsuf, usuf in zip(_DOWN, _UP):
            if not key.endswith(dsuf):
                continue
            prefix = key[: -len(dsuf)]
            ukey, akey, dora = prefix + usuf, prefix + ".alpha", prefix + ".dora_scale"
            d, u = tensors[key], tensors.get(ukey)
            if u is None or dora in tensors or d["dtype"] not in _FLOATS | {"F64"} or u["dtype"] not in _FLOATS | {"F64"}:
                continue
            ds, us = d["shape"], u["shape"]
            r = ds[0]
            # up must be (out, r) or (out, r, 1, 1); down (r, in) or (r, in, kh, kw)
            if len(ds) < 2 or us[1] != r or (len(us) > 2 and any(x != 1 for x in us[2:])):
                continue
            mods.append({"prefix": prefix, "down": key, "up": ukey, "alpha": akey if akey in tensors else None,
                         "rank": r, "in": int(np.prod(ds[1:])), "out": us[0], "dshape": ds, "ushape": us})
            used.update([key, ukey] + ([akey] if akey in tensors else []))
    return mods, used


def _decompose(tensors: dict, m: dict):
    """-> (Qu, Qd, U, S, Vt) with  delta = Qu @ U @ diag(S) @ Vt @ Qd.T"""
    down = _to_f32(tensors[m["down"]]).reshape(m["rank"], -1).astype(np.float64)
    up = _to_f32(tensors[m["up"]]).reshape(m["out"], m["rank"]).astype(np.float64)
    alpha = float(_to_f32(tensors[m["alpha"]]).reshape(-1)[0]) if m["alpha"] else None
    scale = alpha / m["rank"] if alpha else 1.0
    qu, ru = np.linalg.qr(up)
    qd, rd = np.linalg.qr(down.T)
    u, s, vt = np.linalg.svd(ru @ rd.T * scale, full_matrices=False)
    return qu, qd, u, s, vt


def _rank_for(s: np.ndarray, mode: str, value: float, cap: int) -> int:
    if mode == "rank":
        return max(1, min(int(value), cap))
    e = np.cumsum(s ** 2)
    if e[-1] <= 0:
        return 1
    k = int(np.searchsorted(e / e[-1], value / 100.0) + 1)
    return max(1, min(k, cap))


def _out_dtype(choice: str, original: str) -> str:
    return {"fp16": "F16", "bf16": "BF16", "fp32": "F32"}.get(choice, original if original in _FLOATS else "F32")


# ------------------------------------------------------------------- analysis

def _plan_size(tensors: dict, mods, ks, dtype_choice: str) -> int:
    """Bytes the file would take with these per-layer ranks."""
    owned = set()
    total = 0
    for m, k in zip(mods, ks):
        od = _out_dtype(dtype_choice, tensors[m["down"]]["dtype"])
        bpe = _BYTES[od]
        total += k * (m["in"] + m["out"]) * bpe + (4 if m["alpha"] else 0)
        owned.update([m["down"], m["up"]] + ([m["alpha"]] if m["alpha"] else []))
    for name, t in tensors.items():
        if name in owned:
            continue
        if t["dtype"] in _FLOATS and dtype_choice in ("fp16", "bf16", "fp32") and not name.endswith(".alpha"):
            total += int(np.prod(t["shape"] or [1])) * _BYTES[_out_dtype(dtype_choice, t["dtype"])]
        else:
            total += len(t["raw"])
    return total + 2048


def analyze(path: str, dtype: str = "keep") -> dict:
    """Preview what each size option would cost, without writing anything."""
    tensors, meta = _read(path)
    mods, used = _modules(tensors)
    if not mods:
        raise SlimError("No standard LoRA layers found in this file (LoHa / LoKr / DoRA files are not supported).")
    spectra = [_decompose(tensors, m)[3] for m in mods]
    orig_size = Path(path).stat().st_size
    max_rank = max(m["rank"] for m in mods)
    energy_total = sum(float((s ** 2).sum()) for s in spectra) or 1.0
    dtypes = sorted({tensors[m["down"]]["dtype"] for m in mods})

    def option(oid, label, mode, value):
        ks = [_rank_for(s, mode, value, m["rank"]) for m, s in zip(mods, spectra)]
        kept = sum(float((s[:k] ** 2).sum()) for s, k in zip(spectra, ks)) / energy_total
        size = _plan_size(tensors, mods, ks, dtype)
        return {"id": oid, "mode": mode, "value": value, "label": label, "size_mb": round(size / 1048576, 1),
                "saved_pct": round(100 * (1 - size / orig_size), 1), "kept_pct": round(100 * kept, 2),
                "avg_rank": round(sum(ks) / len(ks), 1)}

    opts = []
    for r in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if r < max_rank:
            opts.append(option(f"rank{r}", f"Rank {r}", "rank", r))
    for e in (99.9, 99, 97, 95, 90):
        opts.append(option(f"energy{e}", f"Smart: keep {e}% of the change", "energy", e))
    opts.append(option("same", "Same rank (only change the number format)", "rank", max_rank))
    # recommendation: the smallest option that still keeps >= 99% of the change
    good = [o for o in opts if o["kept_pct"] >= 99.0 and o["id"] != "same"]
    best = min(good, key=lambda o: o["size_mb"])["id"] if good else None
    return {
        "path": path, "size_mb": round(orig_size / 1048576, 1), "layers": len(mods), "max_rank": max_rank,
        "dtype": "/".join({"F32": "fp32", "F16": "fp16", "BF16": "bf16", "F64": "fp64"}.get(d, d) for d in dtypes),
        "skipped": len([k for k in tensors if k not in used and not k.endswith(".alpha")]),
        "base_model": meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or "",
        "options": opts, "recommended": best,
    }


# ------------------------------------------------------------------------ run

def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not cand.exists():
            return cand
    raise SlimError("Could not find a free output file name.")


def slim(path: str, mode: str = "energy", value: float = 99.0, dtype: str = "keep", out_path: str = "") -> dict:
    """Write a smaller copy of the LoRA. The original file is never touched."""
    if mode not in ("rank", "energy"):
        raise SlimError("mode must be 'rank' or 'energy'.")
    if not (value > 0):
        raise SlimError("Give a rank above 0, or a percentage between 1 and 100.")
    if mode == "energy" and value > 100:
        raise SlimError("The percentage must be 100 or less.")
    tensors, meta = _read(path)
    mods, used = _modules(tensors)
    if not mods:
        raise SlimError("No standard LoRA layers found in this file (LoHa / LoKr / DoRA files are not supported).")
    src = Path(path)
    orig_size = src.stat().st_size

    out_tensors: dict = {}
    kept_e = tot_e = 0.0
    changed = 0
    worst = []
    group_e: dict[str, list[float]] = {}
    for m in mods:
        qu, qd, u, s, vt = _decompose(tensors, m)
        k = _rank_for(s, mode, value, m["rank"])
        e_all, e_keep = float((s ** 2).sum()), float((s[:k] ** 2).sum())
        kept_e += e_keep
        tot_e += e_all
        grp = "text encoder" if "lora_te" in m["prefix"] or "text_model" in m["prefix"] else "model"
        g = group_e.setdefault(grp, [0.0, 0.0])
        g[0] += e_keep
        g[1] += e_all
        od = _out_dtype(dtype, tensors[m["down"]]["dtype"])
        if k >= m["rank"]:
            # nothing to cut here: keep the original numbers exactly (only the number format may change)
            for key in (m["down"], m["up"]):
                t = tensors[key]
                out_tensors[key] = t if od == t["dtype"] else {"dtype": od, "shape": t["shape"], "raw": _from_f32(_to_f32(t), od)}
            if m["alpha"]:
                out_tensors[m["alpha"]] = tensors[m["alpha"]]
            continue
        changed += 1
        rs = np.sqrt(s[:k])
        new_up = (qu @ (u[:, :k] * rs)).astype(np.float32)                    # out x k
        new_down = ((rs[:, None] * vt[:k]) @ qd.T).astype(np.float32)         # k x in
        dshape = [k] + list(m["dshape"][1:])
        ushape = [m["out"], k] + list(m["ushape"][2:])
        out_tensors[m["down"]] = {"dtype": od, "shape": dshape, "raw": _from_f32(new_down.reshape(dshape), od)}
        out_tensors[m["up"]] = {"dtype": od, "shape": ushape, "raw": _from_f32(new_up.reshape(ushape), od)}
        if m["alpha"]:   # the alpha/rank scale is already folded in, so alpha == rank
            out_tensors[m["alpha"]] = {"dtype": "F32", "shape": [], "raw": struct.pack("<f", float(k))}
        rel = float(np.sqrt(max(0.0, 1 - e_keep / e_all))) if e_all > 0 else 0.0
        worst.append((rel, m["prefix"], m["rank"], k))

    for name, t in tensors.items():           # everything we did not touch is copied over
        if name in out_tensors or name in used:
            continue
        if t["dtype"] in _FLOATS and dtype in ("fp16", "bf16", "fp32"):
            od = _out_dtype(dtype, t["dtype"])
            t = t if od == t["dtype"] else {"dtype": od, "shape": t["shape"], "raw": _from_f32(_to_f32(t), od)}
        out_tensors[name] = t

    ranks = [out_tensors[m["down"]]["shape"][0] for m in mods]
    meta = {k: v for k, v in meta.items() if k not in ("sshs_model_hash", "sshs_legacy_hash")}   # hashes no longer match
    meta["ss_network_dim"] = str(max(ranks))
    if any(m["alpha"] for m in mods):
        meta["ss_network_alpha"] = str(max(ranks))
    meta["slimmed_from"] = f"{src.name} (rank {max(m['rank'] for m in mods)} -> max {max(ranks)})"

    if out_path:
        dest = Path(out_path).expanduser()
        if dest.is_dir():
            dest = dest / f"{src.stem}_slim.safetensors"
        if dest.suffix.lower() != ".safetensors":
            dest = dest.with_suffix(".safetensors")
        if dest.resolve() == src.resolve():
            raise SlimError("The output file must be different from the original.")
    else:
        dest = src.with_name(f"{src.stem}_slim_r{max(ranks)}.safetensors")
    dest = _unique(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write(str(dest), out_tensors, meta)
    except OSError:
        try:
            dest.unlink()               # don't leave a half-written .safetensors behind
        except OSError:
            pass
        raise

    new_size = dest.stat().st_size
    worst.sort(reverse=True)
    return {
        "out_path": str(dest), "size_before_mb": round(orig_size / 1048576, 2), "size_after_mb": round(new_size / 1048576, 2),
        "saved_pct": round(100 * (1 - new_size / orig_size), 1),
        "kept_pct": round(100 * kept_e / tot_e, 2) if tot_e else 100.0,
        "layers": len(mods), "layers_changed": changed, "rank_before": max(m["rank"] for m in mods), "rank_after": max(ranks),
        "groups": {g: round(100 * v[0] / v[1], 2) if v[1] else 100.0 for g, v in group_e.items()},
        "worst": [{"layer": p, "rank_before": rb, "rank_after": ra, "error_pct": round(100 * rel, 2)} for rel, p, rb, ra in worst[:5]],
    }


# ------------------------------------------------------------------ metadata

def _digest(tensors: dict) -> dict:
    """sha256 of every tensor's raw bytes - used to prove a copy is intact."""
    return {name: (t["dtype"], list(t["shape"]), hashlib.sha256(t["raw"]).hexdigest())
            for name, t in tensors.items()}


def metadata(path: str) -> dict:
    """The `__metadata__` block of a .safetensors file, without reading tensors."""
    p = Path(path)
    if not p.is_file():
        raise SlimError(f"File not found: {path}")
    if p.suffix.lower() != ".safetensors":
        raise SlimError("Pick a .safetensors file.")
    if p.stat().st_size > MAX_BYTES:
        raise SlimError("That file is larger than 3 GB - it does not look like a LoRA.")
    with open(p, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise SlimError("That file is not a valid .safetensors file.")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 * 1024 ** 2:
            raise SlimError("That file is not a valid .safetensors file.")
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {}) or {}
    keys = sorted(({"key": str(k), "value": str(v), "size": len(str(v))} for k, v in meta.items()),
                  key=lambda d: d["key"])
    return {
        "path": str(p), "name": p.name, "size_mb": round(p.stat().st_size / 1048576, 2),
        "tensors": len(header), "has_metadata": bool(meta),
        "keys": keys, "key_count": len(keys),
    }


def strip_metadata(path: str, keep: list[str] | None = None, out_path: str = "",
                   dry_run: bool = False) -> dict:
    """Write a copy of `path` with its `__metadata__` block removed.

    `keep` names the metadata keys to keep; everything else goes. With
    `keep=None` (the default) the whole block is dropped. Weights are never
    re-encoded, and the copy is read back and hash-checked before this returns.
    The original file is never modified.
    """
    src = Path(path)
    if not src.is_file():
        raise SlimError(f"File not found: {path}")
    if src.suffix.lower() != ".safetensors":
        raise SlimError("Pick a .safetensors file.")

    tensors, meta = _read(path)
    if not meta:
        raise SlimError("This file has no metadata to remove.")
    keep_set = None if keep is None else {str(k) for k in keep}
    removed = sorted(k for k in meta if keep_set is None or k not in keep_set)
    new_meta = {k: v for k, v in meta.items() if keep_set is not None and k in keep_set}

    result = {
        "path": str(src), "name": src.name,
        "size_before_mb": round(src.stat().st_size / 1048576, 2),
        "keys_before": len(meta), "keys_after": len(new_meta),
        "removed": removed, "kept": sorted(new_meta), "tensors": len(tensors),
    }
    if dry_run:
        return {**result, "dry_run": True}

    if out_path:
        dest = Path(out_path).expanduser()
        if dest.is_dir():
            dest = dest / f"{src.stem}_nometa.safetensors"
        if dest.suffix.lower() != ".safetensors":
            dest = dest.with_suffix(".safetensors")
        if dest.resolve() == src.resolve():
            raise SlimError("The output file must be different from the original.")
    else:
        dest = src.with_name(f"{src.stem}_nometa.safetensors")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = _unique(dest)

    before = _digest(tensors)
    _write(str(dest), tensors, new_meta)
    try:
        after = _digest(_read(str(dest))[0])
    except (SlimError, OSError, ValueError) as exc:      # unreadable output: never leave it behind
        dest.unlink(missing_ok=True)
        raise SlimError(f"The copy could not be verified ({exc}) - nothing was written.") from exc
    if after != before:
        dest.unlink(missing_ok=True)
        raise SlimError("The copy did not match the original - nothing was written.")

    result.update({
        "out_path": str(dest), "size_after_mb": round(dest.stat().st_size / 1048576, 2),
        "verified": True, "has_metadata_after": bool(metadata(str(dest))["has_metadata"]),
    })
    return result
