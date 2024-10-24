import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import folium
from streamlit_folium import st_folium
import requests
from PIL import Image
from io import BytesIO

# Load data
accel_data = pd.read_csv("https://raw.githubusercontent.com/Tumbsi/fysiikanloppuProject/refs/heads/main/Linear%20Acceleration.csv")
gps = pd.read_csv("https://raw.githubusercontent.com/Tumbsi/fysiikanloppuProject/refs/heads/main/Location.csv")

# Remove idle time from the data
idle_time = 100  # seconds
accel_data = accel_data[accel_data['Time (s)'] > idle_time].reset_index(drop=True)
gps = gps[gps['Time (s)'] > idle_time].reset_index(drop=True)

# Streamlit sliders for order and cutoff
order = st.slider("Order value", 1, 10, value=10)
cutoff = st.slider("Cutoff value", 0.3, 10.0, value=1.46)

# Lowpass filter function
def butter_lowpass_filter(data, cutoff, fs, order):
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Apply filter to acceleration data
fs = 50  # Sampling rate
accel_data['filtered'] = butter_lowpass_filter(accel_data['Linear Acceleration y (m/s^2)'], cutoff, fs, order)

# Step count using peak detection
positive_peaks, _ = find_peaks(accel_data['filtered'], height=np.mean(accel_data['filtered']) + 0.5, distance=fs * 0.5)
negative_peaks, _ = find_peaks(-accel_data['filtered'], height=np.mean(-accel_data['filtered']) + 0.5, distance=fs * 0.5)
step_count_filtered = len(positive_peaks) + len(negative_peaks)

# GPS data processing
gps['coords'] = gps.apply(lambda row: (row['Latitude (°)'], row['Longitude (°)']), axis=1)
gps = gps[gps['Horizontal Accuracy (m)'] < 30]  # Filter GPS points with accuracy < 30 meters

# Distance calculation from GPS data
R = 6371000
a = np.sin(np.radians(gps['Latitude (°)']).diff() / 2)**2 + np.cos(np.radians(gps['Latitude (°)'])) * np.cos(np.radians(gps['Latitude (°)']).shift()) * np.sin(np.radians(gps["Longitude (°)"]).diff() / 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
gps['Distance (m)'] = R * c
total_distance = gps['Distance (m)'].sum()
total_time = (gps['Time (s)'].iloc[-1] - gps['Time (s)'].iloc[0])  # in seconds
average_speed = total_distance / total_time  # meters per second

# Step length calculation
step_length = total_distance / step_count_filtered if step_count_filtered > 0 else 0

# Add custom CSS to control layout and reduce padding
st.markdown("""
    <style>
        .section-header {
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .section-content {
            margin-top: 0;
            padding-top: 0;
        }
        .map-section {
            margin-bottom: 10px; /* Reduce space after the map */
        }
        .results-section {
            margin-top: -10px; /* Reduce space between map and results */
        }
        .assessment-section {
            margin-top: -15px; /* Reduce space between results and assessment */
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.title("Acceleration and GPS Data Analysis")

# GPS route on map with results directly below
st.subheader("Route on Map")
lat_mean = gps['Latitude (°)'].mean()
long_mean = gps['Longitude (°)'].mean()
mymap = folium.Map(location=[lat_mean, long_mean], zoom_start=14)

# Polyline for route
df = gps[['Latitude (°)', 'Longitude (°)', 'Horizontal Accuracy (m)', 'Vertical Accuracy (m)']].copy()
for i in range(len(df) - 1):
    start_coords = df.iloc[i][['Latitude (°)', 'Longitude (°)']].values
    end_coords = df.iloc[i+1][['Latitude (°)', 'Longitude (°)']].values
    folium.PolyLine([start_coords, end_coords], color='blue', weight=2.5, opacity=1).add_to(mymap)
st_folium(mymap, width=700, height=500)

# Display results directly below the map
st.subheader("Results", class_="section-header")
st.write(f"Step count (filtered): {step_count_filtered}", class_="section-content")
st.write(f"Step count (Fourier): {step_count_fourier:.2f}")
st.write(f"Average speed: {average_speed:.2f} m/s")
st.write(f"Total distance: {total_distance:.2f} m")
st.write(f"Step length: {step_length:.2f} m")

# Headline for "My assessment"
st.subheader("My Assessment", class_="section-header assessment-section")
st.write(f"Total distance roughly actually walked: ~350m")
st.write(f"My assessment is that the calculations are very accurate for the data taken.")
st.write(f"Conclusion is that my phone is on its last straw, this is the best it can do.")
st.write(f"My walk was brisk and I was walking at a steady pace, so I believe the average speed and step length are accurate!")

# Image section directly below the assessment
url = "https://github.com/Tumbsi/fysiikanloppuProject/blob/5ea374d291475288af8786ca89c96e2e8183d8ce/actualtravel.png?raw=true"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
st.markdown("<hr>", unsafe_allow_html=True)  # Use horizontal line to separate content
st.image(image, caption='Real path taken, roughly 350m', use_column_width=True)
