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

def descriptions(prediction_dict):
    long_description = []

    temp = prediction_dict.get("feelslikemax c", 0)
    humidity = prediction_dict.get("humidity", 0)
    cloudcover = prediction_dict.get("cloudcover", 0)
    precipcover = prediction_dict.get("precipcover", 0)
    rainfall = prediction_dict.get("precip", 0)
    windspeed = prediction_dict.get("windspeed", 0)

    # --- RAIN PRIORITY ---
    if precipcover >= 70 and rainfall >= 10:
        shortdescription=  "Heavy widespread rainfall"
        long_description.extend("Expect heavy rainfall across the region.")
        bgImage = "overcast"
    if precipcover >= 50 and rainfall >= 5:
        shortdescription=  "Widespread rain expected"
        long_description.append("Widespread rain is anticipated across large regions of this area.")
    if precipcover >= 50 and rainfall < 5:
        shortdescription=  "Scattered light showers"
        long_description.append("Light showers are possible throughout the day.")
        bgImage = "rain"
    if precipcover < 50 and rainfall >= 5:
        shortdescription=  "Isolated showers expected"
        long_description.append("Isolated showers are expected in some areas.")
        bgImage = "rain"
    if precipcover < 50 and rainfall < 5 and rainfall > 0:
        shortdescription=  "Light rain possible"
        long_description.append("Light rain is possible, but amounts are expected to be minimal.")
        bgImage = "rain"

    # --- HEAT PRIORITY ---
    if temp >= 35 and humidity >= 70:
        shortdescription=  "Dangerously hot conditions"
        long_description.append("Extreme heat combined with high humidity may lead to heat-related illnesses. Stay hydrated and avoid strenuous outdoor activities.")
        bgImage = "sunny"
    if temp >= 30:
        shortdescription=  "Generally hot weather"
        long_description.append("Generally hot weather is expected. Stay hydrated and avoid strenuous outdoor activities.")
        bgImage = "sunny"

    if humidity >= 75:
        shortdescription=  "Oppressive humidity"
        long_description.append("High humidity levels may cause discomfort. Take precautions to stay cool.")
        bgImage = "humid"


    # --- CLOUD PRIORITY ---
    if cloudcover >= 70:
        shortdescription=  "Overcast cloudy skies"
        long_description.append("Expect overcast conditions with limited sunshine throughout the day.")
        bgImage = "overcast"
    if cloudcover < 40:
        shortdescription=  "Partly cloudy skies"
        long_description.append("Partly cloudy skies are expected throughout the day.")
        bgImage = "cloudy"
    if cloudcover < 20:
        shortdescription=  "Mostly sunny skies"
        long_description.append("Mostly sunny skies are expected throughout the day.")

    # --- WIND CHECK (optional but important) ---
    if windspeed >= 30:
        shortdescription=  "Strong windy conditions"
        long_description.append("Strong winds are expected. Secure loose objects and be cautious while driving.")

    # --- FALLBACK ---
    if not shortdescription:
        shortdescription =  "A pleasant day"
        long_description.append("Overall, a pleasant day with mild weather conditions. Perfect for outdoor activities, or simply relaxing with family and friends.")
    

    return shortdescription, " ".join(long_description), bgImage

def warning_check(prediction_dict):
    long_warning = []
    warnings = []  
    # Check precipitation warnings
    precip = prediction_dict.get("precip")
    if precip is not None and precip < 0:
        precip = 0  # Ensure precip is not negative
        prediction_dict["precip"] = 0
    if precip > 20:
        warnings.append("Severe Flooding possible")
      
    elif precip > 10:
        warnings.append("Flooding Warning")
       
    elif precip > 5:
        warnings.append("Flooding Advisory")
       
    elif precip == 0:
        warnings.append("No Rainfall Expected")



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
        warnings.append("Wind Warning")
     
    elif windspeed > 10:
        warnings.append("Wind Advisory")
       

    # If no warnings were added, return "No Warnings"
    if not warnings:
        warnings.append("No Warnings")

    return warnings

def convert_seconds_to_hours_minutes(seconds):
    hours = int(seconds // 3600) % 12

    minutes = int ((seconds % 3600) // 60)

    time = f"{hours:02d}:{minutes:02d}"
    return time

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
    
    try:
        location_encoder = joblib.load("WeatherModel/Models/label_encoder.pkl")
        models = {}
        for target in target_features:
            models[target] = joblib.load(f"WeatherModel/models/temp_based/{target}_model.pkl")

        location_encoded = int(location)
        week_predictions = []
        week_warnings = []
        for i in range(7):
            dt = datetime.strptime(date, "%d-%m-%Y") + pd.DateOffset(days=i)
            input_dataf = prepare_input_from_date(dt.strftime("%d-%m-%Y"), location_encoded)
            week_preds = predict_for_date(input_dataf, models, target_features)
            week_prediction_dict = {target: week_preds[target][0] for target in target_features}

            # Set precip to 0 if negative
            if week_prediction_dict.get("precip") is not None and week_prediction_dict["precip"] < 0:
                week_prediction_dict["precip"] = 0

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
               "precipcover": week_prediction_dict.get("precipcover"),
                "sunrise": convert_seconds_to_hours_minutes(week_prediction_dict.get("sunrise")),
                "sunset": convert_seconds_to_hours_minutes(week_prediction_dict.get("sunset")),
                "winddir": week_prediction_dict.get("winddir"),
                "warnings": week_warnings[-1]
            })

            warnings= week_warnings[0]
            short_description, long_description, bgImage = descriptions(week_predictions[0])

        for target in target_features:
            del models[target]

        location = location_encoder.inverse_transform([int(location_encoded)])[0]

        return render_template('result.html', location=location, date=date, warnings=warnings, predictions=week_predictions, descriptions=long_description, short_descriptions=short_description, image= bgImage), 200

    except Exception as e:
        error_message = f"Error during result rendering: {str(e)}"
        return render_template('error.html', error_message=error_message), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080, debug=True)