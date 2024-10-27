import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
from logging import Logger
import logging
from typing import List
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
from app.weather_API import *
import os
import json
import pandas as pd
from dotenv import load_dotenv
from models import PipelineRequestModel
from numpy import ndarray
import numpy as np
import shutil
import cv2
import tempfile
import torch
import torchvision.transforms as transforms
from collections import Counter
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import time
from PIL import Image


# Setup Open-Meteo API with caching and retries
def setup_openmeteo_client():
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        return openmeteo
    except Exception as e:
        print(f"Error setting up Open-Meteo client: {e}")
        return None

# Fetch weather data from Open-Meteo API
def fetch_weather_data(latitude, longitude, start_date, end_date):
    try:
        openmeteo = setup_openmeteo_client()
        if openmeteo is None:
            return None

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(["rain", "snowfall", "cloud_cover", "sunshine_duration"])
        }

        responses = openmeteo.weather_api(url, params=params)
        return responses[0] if responses else None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def process_weather_data(response):
    try:
        hourly = response.Hourly()
        if hourly is None:
            print("No hourly data found.")
            return pd.DataFrame()
        
        cloud_cover = hourly.Variables(0).ValuesAsNumpy() if hourly.Variables(0) else None
        rain = hourly.Variables(1).ValuesAsNumpy() if hourly.Variables(1) else None
        snowfall = hourly.Variables(2).ValuesAsNumpy() if hourly.Variables(2) else None
        sunshine_duration = hourly.Variables(3).ValuesAsNumpy() if hourly.Variables(3) else None

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        if cloud_cover is not None:
            hourly_data["cloud_cover"] = cloud_cover
        if rain is not None:
            hourly_data["rain"] = rain
        if snowfall is not None:
            hourly_data["snowfall"] = snowfall
        if sunshine_duration is not None:
            hourly_data["sunshine_duration"] = sunshine_duration

        return pd.DataFrame(data=hourly_data)
    except Exception as e:
        print(f"Error processing weather data: {e}")
        return pd.DataFrame()

# Determine the weather condition based on mean values
def get_weather_condition(df):
    if df.empty:
        return None

    avg_cloud_cover = df["cloud_cover"].mean() if "cloud_cover" in df.columns else 0
    avg_rain = df["rain"].mean() if "rain" in df.columns else 0
    avg_snowfall = df["snowfall"].mean() if "snowfall" in df.columns else 0
    avg_sunshine = df["sunshine_duration"].mean() if "sunshine_duration" in df.columns else 0

    # Simple threshold-based logic to determine weather condition
    if avg_rain > 0.1:  # Example threshold for rain
        return "Rainy"
    elif avg_snowfall > 0.1:  # Example threshold for snow
        return "Snowy"
    elif avg_cloud_cover < 20 and avg_sunshine > 0:  # Mostly clear and sunny
        return "Sunny"
    elif avg_cloud_cover >= 50:  # Overcast or partly cloudy
        return "Cloudy"
    else:
        return "Partly Cloudy"



'''IMAGE AUGMENTATIONS---------------------------------------------------------------------------------------------------------------------------'''
# Improved weather classification
def process_and_classify_weather(image_path, model, grid_size=3, confidence_threshold=0.6):
    try:
        # Load and split the image into grids
        image = Image.open(image_path)
        grids = split_image(image, grid_size)

        predictions = []
        last_weather = None

        # Process and classify each grid
        for grid in grids:
            image_tensor = preprocess_image(grid)
            if image_tensor is None:
                continue
            
            predicted_class, confidence = classify_weather_with_confidence(image_tensor, model)
            
            # Map class to weather type if above confidence threshold
            weather_class = map_to_weather_class(predicted_class, last_weather) if confidence >= confidence_threshold else "Unknown Weather"
            
            if weather_class != "Unknown Weather":
                last_weather = weather_class
            predictions.append(weather_class)

        # Get the most common weather type
        filtered_predictions = [weather for weather in predictions if weather != "Unknown Weather"]
        return Counter(filtered_predictions).most_common(1)[0][0] if filtered_predictions else "Unknown"
    
    except Exception as e:
        print(f"Error processing image for weather classification: {e}")
        return "Unknown"

# Preprocess image for model input
def preprocess_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to split the image into grids
def split_image(image, grid_size):
    try:
        width, height = image.size
        grid_width = width // grid_size
        grid_height = height // grid_size
        return [image.crop((i * grid_width, j * grid_height, (i + 1) * grid_width, (j + 1) * grid_height))
                for i in range(grid_size) for j in range(grid_size)]
    except Exception as e:
        print(f"Error splitting image: {e}")
        return []

# Classify with confidence score
def classify_weather_with_confidence(image_tensor, model):
    try:
        with torch.no_grad():
            outputs = model(image_tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = probabilities.max(1)
        return preds.item(), confidence.item()
    except Exception as e:
        print(f"Error during weather classification: {e}")
        return None, 0

# Function to map ImageNet class to weather type
def map_to_weather_class(imagenet_class, last_weather=None):
    weather_mapping = {
        # Cloudy/Overcast
        "cloud": "cloudy",
        "overcast": "cloudy",
        "fog": "foggy",
        "mist": "foggy",
        "alp": "cloudy",
        "geyser": "cloudy",
        "mountain": "cloudy",
        "iceberg": "cloudy",
        "volcano": "cloudy",
        
        # Rain-related
        "umbrella": "rainy",
        "rain": "rainy",
        "poncho": "rainy",
        "fountain": "rainy",
        "waterfall": "rainy",
        "gown": "rainy",
        "raincoat": "rainy",
        "rainy": "rainy",

        # Snow-related
        "snowplow": "snowy",
        "ski": "snowy",
        "sled": "snowy",
        "igloo": "snowy",
        "snowmobile": "snowy",
        "snowman": "snowy",
        "ice_skater": "snowy",
        "snowy": "snowy",
        
        # Storm/Extreme Weather
        "parachute": "stormy",
        "storm": "stormy",
        "thunderstorm": "stormy",
        "missile": "stormy",
        "tornado": "stormy",
        "cyclone": "stormy",
        "fire": "stormy",
        "tsunami": "stormy",
        "stormy": "stormy",

        # Sunny/Clear Weather
        "sun": "sunny",
        "beach": "sunny",
        "sandbar": "sunny",
        "solar_dish": "sunny",
        "sunglasses": "sunny",
        "sunscreen": "sunny",
        "lakeside": "sunny",
        "pier": "sunny",
        "shadow" : "sunny",
        "sunny": "sunny",

        # Foggy Weather
        "lighthouse": "foggy",
        "foggy": "foggy",

        # Generic Nature
        "water": "rainy",
        "tree": "cloudy",
        "plant": "cloudy",
        "forest": "cloudy",
        "bush": "cloudy",
        "field": "cloudy",
        "cloudy": "cloudy",
        
        # Edge Cases
        "Unknown": "Unknown Weather"
    }

    weather_class = weather_mapping.get(imagenet_class, "Unknown Weather")
    if weather_class == "Unknown Weather" and last_weather:
        return last_weather
    else:
        return weather_class

'''HELPER FUNCTIONS------------------------'''

def format_date(date_str):
    try:
        # Parse the date in "YY-MM-DD" format
        parsed_date = datetime.strptime(date_str, "%y-%m-%d")
        # Format it to "YYYY-MM-DD"
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"Date formatting error: {e}")
        return None


def extract_latitude_longitude_from_filename(filename):
    try:
        # "Webcam__35.652832_139.839478_15_10_24_1729185796.5170758.png" example filename
        split_filename = filename.split("_")
        latitude = split_filename[2]
        longitude = split_filename[3]
    
        return latitude, longitude
    except Exception as e:
        print(f"error : {e} , Check Image filename")
    
def extract_date_from_filename(filename):
    try:
        # "Webcam__35.652832_139.839478_15_10_24_1729185796.5170758.png"  example filename
        split_filename = filename.split("_")
        year = split_filename[6]
        month = split_filename[5]
        day = split_filename[4]
    
        return f"{year}-{month}-{day}"
    except Exception as e:
        print(f"error : {e} , Check Image filename")
        


def save_results_to_json(url, date, latitude, longitude, weather, filename='results.json'):
    try:
        result = {
            "type": "Combined Dataset",
            "weather": weather,
            "date": date,
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude
            }
        }

        try:
            with open(filename, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(result)

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
