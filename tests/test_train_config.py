"""UI state -> backend payload. These catch settings that silently never reach the trainer."""
import copy

from app import train_config as tc


def _flat(d, out=None):
    out = {} if out is None else out
    for k, v in d.items():
        if isinstance(v, dict):
            _flat(v, out)
        else:
            out[k] = v
    return out


def _args(ui):
    return _flat(tc.build_backend_payload(ui)["args"])


def test_default_payload_builds():
    ui = tc.default_ui_payload()
    payload = tc.build_backend_payload(ui)
    assert {"args", "dataset", "subsets"} <= set(payload)


def test_learning_rate_and_network_reach_args():
    ui = tc.default_ui_payload()
    ui["optimizer"]["learning_rate"] = "3e-5"
    ui["network"]["network_dim"] = 24
    ui["network"]["network_alpha"] = 12
    a = _args(ui)
    assert str(a["learning_rate"]) in ("3e-05", "3e-5", "0.00003")
    assert a["network_dim"] == 24 and a["network_alpha"] == 12


def test_logging_on_by_default_and_tensorboard():
    a = _args(tc.default_ui_payload())
    assert a.get("log_with") == "tensorboard"


def test_logging_off_sends_no_log_args():
    ui = tc.default_ui_payload()
    ui["logging"]["enabled"] = False
    assert "log_with" not in _args(ui)


def test_sample_prompts_text_is_forwarded():
    ui = tc.default_ui_payload()
    ui["sample"]["enabled"] = True
    ui["sample"]["prompts_text"] = "1girl, solo --w 832 --h 1216"
    a = _args(ui)
    assert a["sample_prompts_text"] == "1girl, solo --w 832 --h 1216"


def test_rex_scheduler_gamma_and_d_are_forwarded():
    ui = tc.default_ui_payload()
    ui["optimizer"]["lr_scheduler"] = "rex_annealing_warm_restarts_(RAWR)"
    ui["optimizer"]["gamma"] = 0.9
    ui["optimizer"]["d_param"] = 0.9
    a = _args(ui)
    assert "RexAnnealingWarmRestarts" in str(a["lr_scheduler_type"])
    joined = " ".join(a["lr_scheduler_args"])
    assert "gamma=" in joined and "d=" in joined


def test_subset_settings_reach_dataset():
    ui = tc.default_ui_payload()
    ui["subsets"] = [{
        "name": "s", "image_dir": "/x", "num_repeats": 3, "caption_extension": ".txt",
        "shuffle_caption": True, "keep_tokens": 1, "flip_aug": True, "random_crop": False,
        "color_aug": False, "is_reg": False, "is_val": False,
    }]
    sub = tc.build_backend_payload(ui)["subsets"][0]
    assert sub["num_repeats"] == 3
    assert sub["shuffle_caption"] is True
    assert sub["keep_tokens"] == 1
    assert sub["flip_aug"] is True


def test_toml_roundtrip_is_stable():
    """UI -> backend payload -> TOML -> UI -> backend payload must give the same payload."""
    ui = tc.default_ui_payload()
    ui["network"]["network_dim"] = 20
    ui["optimizer"]["learning_rate"] = "2e-4"
    payload = tc.build_backend_payload(ui)
    doc = tc.to_toml_document(payload)
    ui2 = tc.to_ui_payload(tc.from_toml_document(doc))
    payload2 = tc.build_backend_payload(ui2)
    assert _flat(payload2["args"]) == _flat(payload["args"])
