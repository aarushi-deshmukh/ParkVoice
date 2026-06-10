import numpy as np

from explainability.clinical_rules import build_biomarker_dashboard
from pipeline.audio_quality import assess_audio_quality
from pipeline.uncertainty import estimate_uncertainty


def test_audio_quality_contract():
    sr = 22050
    tone = 0.2 * np.sin(2 * np.pi * 180 * np.arange(sr * 4) / sr)
    result = assess_audio_quality(tone.astype(np.float32), sr)

    assert set(["quality_score", "quality_category", "warnings"]).issubset(result)
    assert result["quality_category"] in {"Excellent", "Good", "Fair", "Poor"}
    assert 0 <= result["quality_score"] <= 1


def test_uncertainty_contract():
    result = estimate_uncertainty(
        {"rf": 0.42, "xgb": 0.47, "lgbm": 0.44, "ensemble": 0.45},
        0.45,
        {"quality_score": 0.8, "quality_category": "Good"},
    )

    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["uncertainty_score"] <= 1
    assert "warnings" in result


def test_biomarker_dashboard_required_rows():
    rows = build_biomarker_dashboard({
        "jitter_local": 1.2,
        "shimmer_local": 3.2,
        "hnr": 19.0,
        "nhr": 0.01,
        "rpde": 0.3,
        "dfa": 0.6,
        "ppe": 0.1,
        "pitch_cv": 0.08,
        "d2": 2.0,
    })

    names = {row["name"] for row in rows}
    assert {"Jitter", "Shimmer", "HNR", "NHR", "RPDE", "DFA", "PPE", "Pitch Variability", "Signal Complexity"} <= names
    assert all(row["status"] in {"Normal", "Borderline", "Abnormal"} for row in rows)
