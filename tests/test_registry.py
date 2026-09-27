import numpy as np
import pytest

from ancdata.fixtures import fake_noise, fake_speech
from ancdata.registry import Row, assign_split, esc50_fold_split, has_voice, load_manifest, validate


def test_split_is_deterministic_and_speaker_disjoint(data_root):
    assert assign_split("spk1") == assign_split("spk1")
    m = load_manifest()
    sp = m[m.sclass == "speech"]
    train = set(sp[sp.split == "train"].speaker_id)
    test = set(sp[sp.split == "test"].speaker_id)
    assert train and test and not (train & test)


def test_esc50_folds():
    assert [esc50_fold_split(f) for f in (1, 2, 3, 4, 5)] == ["train"] * 3 + ["val", "test"]
    with pytest.raises(ValueError):
        esc50_fold_split(6)


def test_manifest_paths_are_relative_posix(data_root):
    m = load_manifest()
    assert not m.path.str.contains("\\\\", regex=False).any()
    assert not m.path.str.startswith("/").any()
    assert not m.path.str.contains(":", regex=False).any()


def test_vad_flags_speechlike_not_wind():
    rng = np.random.default_rng(0)
    assert has_voice(fake_speech(rng, 3.0, 2))
    assert not has_voice(fake_noise(rng, "wind", 3.0))


def _row(**kw) -> Row:
    base = dict(path="x.wav", pile="esc50", sclass="ambience", category="engine", duration_s=5.0,
                sample_rate_native=44100, speaker_id=None, fold=1, crest_factor_db=10.0, peak=0.5,
                has_voice=False, licence="x", split="train", eval_only=False, holdout_group=None)
    base.update(kw)
    return Row(**base)


def test_validate_raises_on_voiced_noise():
    with pytest.raises(ValueError):
        validate(_row(has_voice=True))


def test_validate_raises_on_speech_without_speaker():
    with pytest.raises(ValueError):
        validate(_row(pile="librispeech", sclass="speech", category="speech", speaker_id=None, fold=None))


def test_all_shipped_configs_load():
    from pathlib import Path
    from ancdata.battlefield import load_battlefield_config
    from ancdata.config import load_config
    for f in Path(__file__).resolve().parent.parent.joinpath("configs").glob("*.yaml"):
        if f.stem == "battlefield_v4":
            from ancdata.battlefield_v4 import load_v4_config
            load_v4_config(f)                   # v4 chain: label catalogue + budget schema
        elif f.stem.startswith("battlefield"):
            load_battlefield_config(f)          # v3 chain: its own schema (6 s, dry_x_agc target)
        else:
            load_config(f)
