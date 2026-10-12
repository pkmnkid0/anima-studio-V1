"""
Dataset loading for Anima Studio's native backend.

Reads the exact same on-disk format the rest of Anima Studio already
uses (images with same-named .txt caption files, optionally repeated
per-subset via num_repeats) - see app/services/datasets.py for the
canonical version of this that the Caption/Edit tabs use. This is a
separate, smaller implementation because this one needs to end up as
tensors for training, not JSON for a web UI - but the on-disk
conventions (comma-or-newline tag separation, .txt/.caption fallback)
are kept consistent with it deliberately.
"""
from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class DatasetError(RuntimeError):
    pass


def read_caption(image_path: Path) -> str:
    for suffix in (".txt", ".caption"):
        p = image_path.with_suffix(suffix)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                return p.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def list_subset_items(subset: dict) -> list[dict]:
    """One subset dict (image_dir, num_repeats, caption_extension, etc.)
    -> a flat list of {"path", "caption", "subset"} entries, with each
    image repeated num_repeats times (the standard LoRA dataset
    convention for weighting one folder's contribution to an epoch)."""
    image_dir = subset.get("image_dir")
    if not image_dir:
        raise DatasetError("Subset has no image_dir set.")
    base = Path(image_dir)
    if not base.is_dir():
        raise DatasetError(f"Not a folder: {image_dir}")

    num_repeats = max(1, int(subset.get("num_repeats", 1)))
    trigger_word = (subset.get("keep_tokens_separator") or "").strip()

    items = []
    for f in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption = read_caption(f)
        entry = {"path": str(f), "caption": caption, "subset": subset.get("name", base.name)}
        items.extend([entry] * num_repeats)

    if not items:
        raise DatasetError(f"No images found in {image_dir}.")
    return items


def build_dataset_items(dataset: dict) -> list[dict]:
    """All subsets in a validated `dataset` dict -> one flat, shuffled-
    at-train-time-ready list of items. Raises DatasetError with a
    specific, actionable message on the first subset that's actually
    wrong, rather than a generic "no images" once everything's merged."""
    subsets = dataset.get("subsets") or []
    if not subsets:
        raise DatasetError("No subsets in dataset config.")
    all_items = []
    for subset in subsets:
        all_items.extend(list_subset_items(subset))
    return all_items


def make_torch_dataset(dataset: dict, resolution: int, tokenize_fn):
    """Returns a torch.utils.data.Dataset yielding (pixel_values, caption)
    pairs resized/cropped to `resolution`. `tokenize_fn` is left as a
    caller-supplied hook rather than baked in here, since SDXL needs two
    tokenizers and Anima needs a completely different one (Qwen3) - this
    module only owns "find the images and captions", not "turn them into
    a specific model's expected input"."""
    import torch
    from torchvision import transforms
    from PIL import Image

    items = build_dataset_items(dataset)
    tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, idx):
            item = items[idx]
            with Image.open(item["path"]) as img:
                img = img.convert("RGB")
                pixel_values = tf(img)
            return pixel_values, item["caption"]

    return _Dataset()
