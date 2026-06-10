# Architecture

ParkVoice AI implements the following production screening pipeline:

1. Dataset Validation
2. Audio Quality Assessment
3. Feature Extraction
4. Calibrated Ensemble Risk Assessment
5. Severity Estimation
6. Explainability
7. Uncertainty Estimation
8. ONNX Edge Inference
9. Benchmark Reporting
10. Executive Dashboard

## Clinical Positioning

Use these phrases:

- Parkinson's Risk Assessment
- Acoustic Screening Support
- AI-Assisted Parkinson's Screening

Avoid diagnostic or disease-confirming claims in product copy.

## Explainability

SHAP values are converted into clinician-friendly biomarker language in `backend/explainability/clinical_rules.py`.

Examples:

- Elevated jitter may indicate reduced vocal fold stability.
- Reduced HNR may indicate increased breathiness.

## Uncertainty

`backend/pipeline/uncertainty.py` combines ensemble disagreement, calibration confidence, and audio quality into confidence and uncertainty outputs.
