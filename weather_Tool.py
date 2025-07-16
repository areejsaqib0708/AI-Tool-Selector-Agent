import requests
import google.generativeai as genai
import subprocess
from API import api

api_key =api()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")
def get_coordinates(city):
    """Fetch latitude and longitude for a city using Open-Meteo's geocoding API."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("results"):
            return data["results"][0]["latitude"], data["results"][0]["longitude"]
    return None, None

def get_weather(lat, lon):
    """Fetch current weather and 3-day forecast from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min,weather_code"
        "&forecast_days=3"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def weather_code_to_description(code):
    """Convert Open-Meteo weather code to description."""
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
    }
    return codes.get(code, "Unknown")

def get_weather_summary(city: str) -> str:
    """Main reusable function to fetch and format weather summary for a city."""
    lat, lon = get_coordinates(city)
    if lat is None or lon is None:
        return f"❌ Could not find coordinates for '{city}'."

    weather_data = get_weather(lat, lon)
    if weather_data is None:
        return "❌ Failed to fetch weather data."

    result = f"📍 **Weather in {city}**\n\n"

    current = weather_data["current"]
    result += f"**Current Temperature:** {current['temperature_2m']}°C\n\n"
    result += f"**Humidity:** {current['relative_humidity_2m']}%\n\n"
    result += f"**Condition:** {weather_code_to_description(current['weather_code'])}\n\n"

    daily = weather_data["daily"]
    result += "📅 **3-Day Forecast:**\n"
    for i in range(3):
        date = daily["time"][i]
        min_temp = daily["temperature_2m_min"][i]
        max_temp = daily["temperature_2m_max"][i]
        condition = weather_code_to_description(daily["weather_code"][i])
        result += f"- {date}: {min_temp}°C to {max_temp}°C, {condition}\n"

    return result

def extract_city_from_prompt(prompt: str) -> str:
    """Use Gemini to extract city name from the user prompt."""
    instruction = f"""
    Extract the city name from the user's weather-related query.

    Examples:
    - "What's the weather in Lahore?" → Lahore
    - "Show me weather for New York" → New York
    - "Islamabad" → Islamabad
    - "Tell me the forecast in Karachi" → Karachi

    Now extract the city from this prompt:
    "{prompt}"

    Only return the city name, nothing else.
    """
    response = model.generate_content(instruction)
    return response.text.strip()

