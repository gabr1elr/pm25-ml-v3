# PM2.5 Air Pollution Prediction (Copernicus API)

This project downloads real PM2.5 data from the Copernicus Atmosphere Monitoring Service (CAMS),
processes it, trains a machine-learning model, and predicts air quality in Bucharest.

https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-forecasts?tab=download

## Pipeline
1. Download PM2.5 from Copernicus
2. Process NetCDF
3. Train ML model
4. Predict future PM2.5

## Installation
pip install -r requirements.txt

## Run

main.py
