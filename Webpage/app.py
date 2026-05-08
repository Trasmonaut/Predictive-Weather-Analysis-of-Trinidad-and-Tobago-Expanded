from flask import Flask, render_template, request, redirect, url_for, flash , app
import joblib
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import threading

# Load environment variables from .env file
load_dotenv()


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')

    LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "weathermodels", "label_encoder.pkl")
    BASE_WEATHER_MODELS_PATH = os.path.join(BASE_DIR, "weathermodels", "temp_based/")

    
    TARGET_FEATURES = [
        'cloudcover',
        'avgtemp c',
        'tempmax c',
        'tempmin c',
        'feelslikemax c',
        'feelslikemin c',
        'avgfeelsliketemp c',
        'humidity',
        'dewpoint c',
        'precipcover',
        'precip',
        'sealevelpressure',
        'solarradiation',
        'solarenergy',
        'sunrise',
        'sunset',
        'visibility',
        'windspeed',
        'winddir'
    ]

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

class Converter:
    @staticmethod
    def convert_seconds_to_hours_minutes(seconds):
        if seconds is None:
            return "00:00"
        hours = int(seconds // 3600) % 12
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"

    @staticmethod
    def convert_date_to_string(date_str):
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        day = dt.day
        if 11 <= day <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return dt.strftime(f"%A %B {day}{suffix}, %Y")

class Weather:
    """
    Singleton-style weather model loader for background preloading.
    """
    _instance = None
    _loaded = False
    _lock = threading.Lock()

    def __new__(cls, config: Config):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Weather, cls).__new__(cls)
                    cls._instance._init_vars()
        return cls._instance

    def _init_vars(self):
        self.config = None
        self.label_encoder = None
        self.models = None
        self._loaded = False

    def load_resources(self, config: Config):
        """Loads encoder and models. Safe to call multiple times; loads only once."""
        with self._lock:
            if self._loaded:
                return
            self.config = config
            self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
            self.models = {}
            for target in config.TARGET_FEATURES:
                model_path = f"{config.BASE_WEATHER_MODELS_PATH}{target}_model.pkl"
                self.models[target] = joblib.load(model_path)
            self._loaded = True

    def ensure_loaded(self, config: Config):
        """Background/load-on-demand safety check."""
        if not self._loaded:
            self.load_resources(config)

    def prepare_input_from_date(self, date, location_encoded):
        dt = datetime.strptime(date, "%d-%m-%Y")
        return pd.DataFrame([{
            'location_encoded': location_encoded,
            'dayofweek': dt.weekday(),
            'dayofyear': dt.timetuple().tm_yday,
            'month': dt.month,
            'year': dt.year
        }])

    def predict_for_date(self, input_df):
        self.ensure_loaded(self.config)
        input_copy = input_df.copy()
        preds = {}
        for target in self.config.TARGET_FEATURES:
            model = self.models[target]
            prediction = model.predict(input_copy)
            preds[target] = prediction
            input_copy[f'pred_{target}'] = prediction
        return preds

    def week_forecast(self, date, location_encoded):
        self.ensure_loaded(self.config)
        week_predictions = []
        week_warnings = []
        for i in range(7):
            dt = datetime.strptime(date, "%d-%m-%Y") + timedelta(days=i)
            input_df = self.prepare_input_from_date(dt.strftime("%d-%m-%Y"), location_encoded)
            week_preds = self.predict_for_date(input_df)
            week_prediction_dict = {target: week_preds[target][0] for target in self.config.TARGET_FEATURES}

            # Set precip to 0 if negative
            if week_prediction_dict.get("precip") is not None and week_prediction_dict["precip"] < 0:
                week_prediction_dict["precip"] = 0.000

            warnings = Weather.warning_check(week_prediction_dict)
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
                "sunrise": Converter.convert_seconds_to_hours_minutes(week_prediction_dict.get("sunrise")),
                "sunset": Converter.convert_seconds_to_hours_minutes(week_prediction_dict.get("sunset")),
                "winddir": week_prediction_dict.get("winddir"),
                "warnings": week_warnings[-1]
            })

        return week_predictions, week_warnings

    @staticmethod
    def descriptions(prediction_dict):
        long_description = []
        shortdescription = ""
        bgImage = "default"

        temp = prediction_dict.get("feelslikemax", 0)
        humidity = prediction_dict.get("humidity", 0)
        cloudcover = prediction_dict.get("cloudcover", 0)
        precipcover = prediction_dict.get("precipcover", 0)
        rainfall = prediction_dict.get("precip", 0)
        windspeed = prediction_dict.get("windspeed", 0)

        # RAIN PRIORITY
        if precipcover >= 70 and rainfall >= 10:
            shortdescription = "Heavy widespread rainfall"
            long_description.append("Expect heavy rainfall across the region.")
            bgImage = "overcast"
        elif precipcover >= 50 and rainfall >= 5:
            shortdescription = "Widespread rain expected"
            long_description.append("Widespread rain is anticipated across large regions of this area.")
            bgImage = "rain"
        elif precipcover >= 50 and rainfall < 5:
            shortdescription = "Scattered light showers"
            long_description.append("Light showers are possible throughout the day.")
            bgImage = "rain"
        elif precipcover < 50 and rainfall >= 5:
            shortdescription = "Isolated showers expected"
            long_description.append("Isolated showers are expected in some areas.")
            bgImage = "rain"
        elif precipcover < 50 and 0 < rainfall < 5:
            shortdescription = "Light rain possible"
            long_description.append("Light rain is possible, but amounts are expected to be minimal.")
            bgImage = "rain"

        # WIND CHECK
        if windspeed >= 30:
            shortdescription = "Strong windy conditions"
            long_description.append("Strong winds are expected. Secure loose objects and be cautious while driving.")

        # HEAT PRIORITY
        if temp >= 35 and humidity >= 70:
            shortdescription = "Dangerously hot conditions"
            long_description.append("Extreme heat combined with high humidity may lead to heat-related illnesses. Stay hydrated and avoid strenuous outdoor activities.")
            bgImage = "sunny"
        elif temp >= 30:
            shortdescription = "Generally hot weather"
            long_description.append("Generally hot weather is expected. Stay hydrated and avoid strenuous outdoor activities.")
            bgImage = "sunny"
        elif humidity >= 75:
            shortdescription = "Oppressive humidity"
            long_description.append("High humidity levels may cause discomfort. Take precautions to stay cool.")
            bgImage = "humid"

        # CLOUD PRIORITY
        if cloudcover >= 70:
            shortdescription = "Overcast cloudy skies"
            long_description.append("Expect overcast conditions with limited sunshine throughout the day.")
            bgImage = "overcast"
        elif cloudcover < 40:
            shortdescription = "Partly cloudy skies"
            long_description.append("Partly cloudy skies are expected throughout the day.")
            bgImage = "cloudy"
        elif cloudcover < 20:
            shortdescription = "Mostly sunny skies"
            long_description.append("Mostly sunny skies are expected throughout the day.")

        # FALLBACK
        if not shortdescription:
            shortdescription = "A pleasant day"
            bgImage = "default"
            long_description.append("Overall, a pleasant day with mild weather conditions. Perfect for outdoor activities, or simply relaxing with family and friends.")

        return shortdescription, " ".join(long_description), bgImage

    @staticmethod
    def warning_check(prediction_dict):
        warnings = []
        precip = prediction_dict.get("precip") or 0
        if precip < 0:
            precip = 0.000
            prediction_dict["precip"] = 0.000
        if precip > 20:
            warnings.append("🔴Severe Flooding possible")
        elif precip > 10:
            warnings.append("🟡Flooding Warning")
        elif precip > 5:
            warnings.append("🔵Flooding Advisory")
        elif precip == 0:
            warnings.append("🟢No Rainfall Expected")

        feelslikemax = prediction_dict.get("feelslikemax c", 0)
        if feelslikemax > 40:
            warnings.append("🔴Severe Heat Warning")
        elif feelslikemax > 35:
            warnings.append("🟡Heat Warning")
        elif feelslikemax > 30:
            warnings.append("🔵Heat Advisory")

        windspeed = prediction_dict.get("windspeed", 0)
        if windspeed > 20:
            warnings.append("🔴Severe Wind Warning")
        elif windspeed > 15:
            warnings.append("🟡Wind Warning")
        elif windspeed > 10:
            warnings.append("🔵Wind Advisory")

        if not warnings:
            warnings.append("🟢No Warnings")
        return warnings

# Removed background preloading. Models will be loaded only when prediction is requested.


def create_app(config_object=Config):

    app.config.from_object(config_object)
    app.secret_key = app.config['SECRET_KEY']

    # No background preloading. Models will be loaded on demand.

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(Exception)
    def handle_exception(e):
        error_message = str(e)
        return render_template('error.html', error_message=error_message), 500

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/terms-and-conditions')
    def terms_and_conditions():
        return render_template('TermsAndConditions.html')

    @app.route('/data-visualization')
    def data_visualization():
        return render_template('coming-soon.html')

    @app.route('/model-showcase')
    def model_showcase():
        return render_template('coming-soon.html')

    @app.route('/get_result', methods=['GET'])
    def go_to_result():
        date_str = request.args.get('date')
        location = request.args.get('location')
        return redirect(url_for('predict', date=date_str, location=location))

    @app.route('/result/date=<date>&location=<location>', methods=['GET'])
    def predict(date, location):
        if not validate_inputs(date, location):
            flash("Invalid inputs. Please ensure the date is in the format dd-mm-yyyy and location is a valid number.")
            return redirect(url_for('index'))

        try:
            weather = Weather(Config)
            weather.ensure_loaded(Config)
            location_encoded = int(location)
            week_predictions, week_warnings = weather.week_forecast(date, location_encoded)
            warnings = week_warnings[0]
            short_description, long_description, bgImage = Weather.descriptions(week_predictions[0])
            convertdate = Converter.convert_date_to_string(date)
            location_name = weather.label_encoder.inverse_transform([location_encoded])[0]

            return render_template('result.html',
                                    location=location_name,
                                    date=convertdate,
                                    warnings=warnings,
                                    predictions=week_predictions,
                                    descriptions=long_description,
                                    short_descriptions=short_description,
                                    image=bgImage), 200

        except Exception as e:
            error_message = f"Error during result rendering: {str(e)}"
            return render_template('error.html', error_message=error_message), 500

    return app

app = create_app()

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )