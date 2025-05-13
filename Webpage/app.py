from flask import Flask, render_template, request
import joblib
import sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/moreinfo')
def moreinfo():
    return render_template('more-info.html')



def prepare_input_from_date(date_str, location_encoded):
    dt = datetime.strptime(date_str, "%d-%m-%Y")
    return pd.DataFrame([{
        'location_encoded': location_encoded,
        'dayofweek': dt.weekday(),
        'dayofyear': dt.timetuple().tm_yday,
        'month': dt.month,
        'year': dt.year
    }])

def predict_for_date(input_df, models, target_order):
    input_copy = input_df.copy()
    preds = {}

    for target in target_order:
        model = models[target]
        prediction = model.predict(input_copy)
        preds[target] = prediction
        input_copy[f'pred_{target}'] = prediction

    return preds

def warning_check(prediction_dict):
    warnings = []  

    # Check precipitation warnings
    precip = prediction_dict.get("precip")
    if precip > 200:
        warnings.append("Severe Flooding possible")
    elif precip > 100:
        warnings.append("Flooding Warning")
    elif precip > 50:
        warnings.append("Flooding Advisory")

    # Check feels-like max temperature warnings
    feelslikemax = prediction_dict.get("feelslikemax c")
    if feelslikemax > 40:
        warnings.append("Severe Heat Warning")
    elif feelslikemax > 35:
        warnings.append("Heat Warning")
    elif feelslikemax > 30:
        warnings.append("Heat Advisory")

    # Check windspeed warnings
    windspeed = prediction_dict.get("windspeed")
    if windspeed > 20:
        warnings.append("Severe Wind Warning")
    elif windspeed > 15:
        warnings.append("Wind Waning")
    elif windspeed > 10:
        warnings.append("Wind Advisory")

    # If no warnings were added, return "No Warnings"
    if not warnings:
        warnings.append("No Warnings")

    return warnings
    
    
@app.route('/result', methods=['POST'])
def predict():
    target_features = ['cloudcover', 'humidity', 'windspeed','feelslikemax c','tempmax c', 'tempmin c', 
                'avgtemp c','feelslikemin c', 'avgfeelsliketemp c','dewpoint c', 'precip','visibility']
    # Load the label encoder
    location_encoder = joblib.load("Models/label_encoder.pkl")

    # Load all models from the Models folder into a dictionary
    models = {}
    
    for target in target_features:
        models[target] = joblib.load(f"Models/{target}_model.pkl")
        print(f"Loaded model for {target}")

    # Load target features
    

    date_str = request.form['date']
    location_encoded = int(request.form['location'])


    # Prepare the input data
    input_dataf = prepare_input_from_date(date_str, location_encoded)
    preds = predict_for_date(input_dataf, models, target_features)
    prediction_dict = {target: preds[target][0] for target in target_features}

    warnings = ", ".join(warning_check(prediction_dict))

    # Predictions for the week

    week_predictions = []
    week_warnings = []
    for i in range(7):
        dt = datetime.strptime(date_str, "%d-%m-%Y") + pd.DateOffset(days=i)
        input_dataf = prepare_input_from_date(dt.strftime("%d-%m-%Y"), location_encoded)
        week_preds = predict_for_date(input_dataf, models, target_features)
        week_prediction_dict = {target: week_preds[target][0] for target in target_features}


        warnings = warning_check(week_prediction_dict)
        week_warnings.append(", ".join(warnings))

        week_predictions.append({
            "date": dt.strftime("%d-%m-%Y"),
            "cloudcover": week_prediction_dict.get("cloudcover"),
            "precip": week_prediction_dict.get("precip"),
            "humidity": week_prediction_dict.get("humidity"),
            "windspeed": week_prediction_dict.get("windspeed"),
            "feelslikemax": week_prediction_dict.get("feelslikemax c"),
            "tempmax": week_prediction_dict.get("tempmax c"),
            "tempmin": week_prediction_dict.get("tempmin c"),
            "avgtemp": week_prediction_dict.get("avgtemp c"),
            "feelslikemin": week_prediction_dict.get("feelslikemin c"),
            "avgfeelslike": week_prediction_dict.get("avgfeelsliketemp c"),
            "dewpoint": week_prediction_dict.get("dewpoint c"),
            "visibility": week_prediction_dict.get("visibility"),
            "warnings": week_warnings[-1]
        })

    location = location_encoder.inverse_transform([int(location_encoded)])[0]
    return render_template('result.html', location=location, date=date_str, warnings = warnings,predictions = week_predictions  ,
        cloudcover=prediction_dict.get("cloudcover"),
        precip=prediction_dict.get("precip"),
        humidity=prediction_dict.get("humidity"),
        windspeed=prediction_dict.get("windspeed"),
        feelslikemax=prediction_dict.get("feelslikemax c"),
        tempmax=prediction_dict.get("tempmax c"),
        tempmin=prediction_dict.get("tempmin c"),
        avgtemp=prediction_dict.get("avgtemp c"),
        feelslikemin=prediction_dict.get("feelslikemin c"),
        avgfeelslike=prediction_dict.get("avgfeelsliketemp c"),
        dewpoint=prediction_dict.get("dewpoint c"),
        visibility=prediction_dict.get("visibility"))

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080, debug=True)