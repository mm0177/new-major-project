from datetime import datetime
from meteostat import Point, Daily
import pandas as pd

# Define key port locations (Lat, Lon)
locations = {
    'China': Point(31.2304, 121.4737),  # Shanghai
    'Finland': Point(60.1699, 24.9384), # Helsinki
    'Germany': Point(53.5511, 9.9937),  # Hamburg
    'USA': Point(34.0522, -118.2437),   # Los Angeles
    'India': Point(19.0760, 72.8777),   # Mumbai
    'Italy': Point(44.4056, 8.9463),    # Genoa
    'France': Point(49.4944, 0.1079)    # Le Havre
}

# Time period: Jan 2023 - Present
start = datetime(2023, 1, 1)
end = datetime(2025, 9, 1)

all_weather = []

print("Fetching weather data...")
for country, location in locations.items():
    try:
        # Fetch daily data
        data = Daily(location, start, end)
        data = data.fetch()
        
        # FIX: Changed 'ME' to 'M' for compatibility
        monthly = data.resample('M').mean()
        
        monthly['country'] = country
        monthly['date'] = monthly.index
        
        # Keep only relevant columns if they exist
        cols = ['date', 'country', 'tavg', 'prcp', 'wspd']
        available_cols = [c for c in cols if c in monthly.columns]
        
        all_weather.append(monthly[available_cols])
        print(f"✅ Fetched data for {country}")
        
    except Exception as e:
        print(f"❌ Error for {country}: {e}")

if all_weather:
    weather_df = pd.concat(all_weather)
    weather_df.to_csv('master_weather_data.csv', index=False)
    print("\nSaved to 'master_weather_data.csv'")
else:
    print("\nNo weather data fetched. Check your internet connection.")