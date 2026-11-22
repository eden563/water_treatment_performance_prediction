# Wastewater Effluent COD Prediction

Machine learning regression project that predicts the chemical oxygen demand (COD) of the water leaving a
wastewater treatment plant from measurements taken earlier in the process. Includes a Streamlit web app.

**Live app:** _add your Streamlit link here_

## Project overview
- **Data:** UCI Water Treatment Plant dataset (527 daily records, one plant in Catalonia, 1990 to 1991)
- **Target:** `DQO-S` (effluent COD, units not documented)
- **Inputs:** 22 measurements from the plant inlet, primary settler and secondary settler
- **Model:** Ridge regression on the log of the target (scikit-learn pipeline with median imputation and scaling)
- **Validation:** chronological 70/15/15 split and time-series cross-validation

## Results (held-out test period)
| | Final model | Baseline (mean) |
|---|---|---|
| MAE | 27.85 | 33.36 |
| R² | 0.131 | -0.001 |

The model beats the baseline, but the signal in the data is weak. The biggest errors are on days with
secondary settler problems, which cannot be predicted from the available measurements.

## Repository structure
```
├── app.py                              # Streamlit app
├── requirements.txt
├── data/water-treatment.data           # dataset (CC BY 4.0)
├── models/
│   ├── water_treatment_model.joblib    # saved pipeline
│   └── model_metadata.json             # feature list, ranges, metrics
└── water_treatment_regression.ipynb
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain, run all cells in the notebook (open it from the repo root folder). This recreates the `models/` folder.

## Limitations
One plant, a short record, small validation and test sets, no documented units. This is an educational
prototype and does not replace laboratory analysis or regulatory monitoring.

## Data source
Poch, M. (1993). *Water Treatment Plant* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5FS4C (CC BY 4.0)
