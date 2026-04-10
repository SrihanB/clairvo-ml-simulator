# Vehicle Acquisition Simulator

This project builds a vehicle-level acquisition prediction model and an interactive simulator for exploring bidding strategies.

## Features
- Logistic regression model predicting acquisition probability
- Aggregated offer-level features
- Streamlit app for interactive simulation
- Outlier detection for risky bids

## How to run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py