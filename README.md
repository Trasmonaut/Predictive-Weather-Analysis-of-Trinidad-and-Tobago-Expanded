# Overview 

This repository is an extension of my Data Science project completed at the University of the West Indies. It focuses on building a predictive weather model for Trinidad and Tobago using historical weather data and advanced machine learning techniques. By leveraging XGBoost and Chained Multioutput Regression, the project delivers highly accurate weather forecasts for 16 regions across the country. The resulting model powers TTWeatherPredict, a web-based application that allows users to input a date and location to receive detailed weather predictions. This tool aims to support better planning and decision-making by providing insights into future weather conditions based on historical trends. Check out the WIKI for more infomration in detail

###
# Introduction
A predictive weather model for Trinidad and Tobago using historical weather data and XGBoost. Historical weather data was fed in on 16 reigions of Trinidad and Tobago into an XGBOOST model to predict what the future weather given a date and reigon might be in Trinidad and Tobago, based off what is was historically.This proejct allows persons to have a refrence of what the weather in future might be, down to specific features/aspects for a given date. This would facilitate better planning of daily activies further in the future.

Data Source - Visual Crossing's Historical Weather data for Trinidad and Tobago

XGBoost and Weather Prediction
This app uses XGBoost to make it's predictions, through Chained Multiouput Regression, where the output from one model is used to influence the output of the one that follows it. The result is a model with an average of 98.47% accuracy, on Time Series data for Trinidad and Tobago.

Weather Prediction app - TTPredict
These models were used to form the back end of a website, where users can input a date and location in Trinidad and Tobago, and be output with a prediction of the weather on that given date.

###
# XGBoost and Weather Prediction
This app uses XGBoost to make it's predictions, through Chained Multiouput Regression, where the output from one model is used to influence the output of the one that follows it. 

###
# Weather Prediction app - TTPredict
These models were used to form the back end of a website, where users can input a date and location in Trinidad and Tobago, and be output with a prediction of the weather on that given date.
