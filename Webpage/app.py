from flask import Flask, render_template, request, redirect, url_for, get_flashed_messages, flash
import joblib
import sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key="Paul_taylor"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/TermsAndConditions')
def TermsAndConditions():
    return render_template('TermsAndConditions.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(Exception)
def handle_exception(e):
    error_message= request.args.get('error_message', str(e))
    return render_template('error.html', error_message=error_message), 500

def prepare_input_from_date(date, location_encoded):
    
    dt = datetime.strptime(date, "%d-%m-%Y")
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

def validate_inputs(date_str, location):
    # Check if date_str is provided and matches dd-mm-yyyy format
    if not date_str:
        return False
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        return False

    # Validate location
    if not location or not location.isdigit():
        return False
    location_int = int(location)
    if location_int < 0 or location_int > 15:
        return False

    return True
   

@app.route('/get_result', methods=['GET'])
def go_to_result():
    date_str = request.args.get('date')
   
    location = request.args.get('location')

    # redirect to the correct format: /result/date=<date>?location=<location> only if valid date
    return redirect(url_for('predict', date=date_str, location=location))
   
    
@app.route('/result/date=<date>&location=<location>', methods=['GET'])
def predict(date,location):

    if validate_inputs(date, location) is not True:
        error_message = "Invalid inputs. Please ensure the date is in the format dd-mm-yyyy and location is a valid number."
        flash(error_message)
        return redirect(url_for('index'))

    target_features=['avgtemp c','tempmax c','tempmin c','feelslikemax c','feelslikemin c','avgfeelsliketemp c',
        'humidity','dewpoint c','precipcover','precip','cloudcover','sealevelpressure','solarradiation',
        'solarenergy','sunrise','sunset','visibility','windspeed','winddir']
    
    # Load the label encoder
    try:
        try:
            location_encoder = joblib.load("WeatherModel/Models/label_encoder.pkl")
        except Exception as e:
            error_message = f"Label encoder file not found: {str(e)}"
            return render_template('error.html', error_message=error_message), 500
        # Load all models from the Models folder into a dictionary
        try:
            models = {}
                
            for target in target_features:
                models[target] = joblib.load(f"WeatherModel/models/temp_based/{target}_model.pkl")
                print(f"Loaded model for {target}")
        except Exception as e:
            error_message = f"Error loading model for {target}: {str(e)}"
            return render_template('error.html', error_message=error_message), 500

        try:
            location_encoded = int(location)
            # Prepare the input data
            input_dataf = prepare_input_from_date(date, location_encoded)
            preds = predict_for_date(input_dataf, models, target_features)
            prediction_dict = {target: preds[target][0] for target in target_features}
        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            return render_template('error.html', error_message=error_message), 500

        warnings = ", ".join(warning_check(prediction_dict))

    # Predictions for the week
        try:
            week_predictions = []
            week_warnings = []
            for i in range(7):
                dt = datetime.strptime(date, "%d-%m-%Y") + pd.DateOffset(days=i)
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
        except Exception as e:
            error_message = f"Error during weekly predictions: {str(e)}"
            return render_template('error.html', error_message=error_message), 500
        

        location = location_encoder.inverse_transform([int(location_encoded)])[0]
        return render_template('result.html', location=location, date=date, warnings = warnings,predictions = week_predictions  ,
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




    except Exception as e:
        error_message = f"Error during result rendering: {str(e)}"
        return render_template('error.html', error_message=error_message), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080, debug=True)