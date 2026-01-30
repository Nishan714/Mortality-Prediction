# Mortality Prediction using Clinical Data and NLP

This repository presents a mortality prediction study using MIMIC-IV data,
comparing a clinical baseline model with an NLP-based model trained on
discharge summaries.

## Methods
- Clinical baseline: age, sex, admission type (logistic regression)
- NLP model: TF-IDF (1–2 grams) + logistic regression
- Patient-level train/validation/test split
- Bootstrap confidence intervals
- Calibration and precision-recall analysis

## Results
The NLP model significantly outperforms the clinical baseline in AUROC and AUPRC.
Calibration and performance plots are provided in the `figures/` directory.

## Important Limitation
Discharge summaries are written after outcomes occur, which introduces label
leakage. Therefore, NLP results represent an upper bound and are not directly
deployable.

## Future Work
- Restrict text to early clinical notes (first 24–48 hours)
- Evaluate transformer-based clinical language models
- External validation on independent datasets

## Reproducibility
This code was developed using Python 3.9.
Install dependencies with:
pip install -r requirements.txt
