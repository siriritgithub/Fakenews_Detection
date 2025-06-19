import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()
weather_key = os.getenv("OPENWEATHER_API_KEY")

# Extract city from sentence
def extract_city(text):
    # Basic extraction: grab the last word that looks like a city name
    city = text.lower().replace("today", "").replace("weather", "").replace("in", "").strip()
    return city.title()

def get_weather(user_input):
    if not weather_key:
        return "⚠️ Weather API key not found."

    city = extract_city(user_input)

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": weather_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("cod") != 200:
        return f"❌ Error: {data.get('message', 'City not found')}"

    temp = data["main"]["temp"]
    condition = data["weather"][0]["description"].capitalize()
    city_name = data["name"]
    return f"📍 {city_name} — 🌡️ {temp}°C | {condition}"
