import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium


accel_data = pd.read_csv("https://raw.githubusercontent.com/Tumbsi/fysiikanloppuProject/refs/heads/main/Linear%20Acceleration.csv")
gps_data = pd.read_csv("https://raw.githubusercontent.com/Tumbsi/fysiikanloppuProject/refs/heads/main/Location.csv")

st.write(f" Default values that gives the most accurate results are: Order = 3, Cutoff = 1.4 & Nyquist = ~21.25")

# idle time to be removed
idle_time = 100  # seconds

# Remove the first 100 seconds of data
accel_data = accel_data[accel_data['Time (s)'] > idle_time].reset_index(drop=True)
gps_data = gps_data[gps_data['Time (s)'] > idle_time].reset_index(drop=True)

# Streamlit sliders for order and cutoff
#order = st.slider("Select order value", 1, 10, value=8)  
#cutoff = st.slider("Select cutoff value", 0.1, 10.0, value=1.5) 
order = st.slider("Order value", 1, 10, value=3)
cutoff = st.slider("Cutoff value", 0.3 , 10.0, value = 1.4)


# lowpass filter
def butter_lowpass_filter(data, cutoff, fs, order):
    nyquist = st.slider("Nyquist frequency", fs/0.5, fs/5, fs/2)
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# Assuming sampling rate is 50 Hz
fs = 50
accel_data['filtered'] = butter_lowpass_filter(accel_data['Linear Acceleration y (m/s^2)'], cutoff, fs, order)

# Step count from filtered data using peak detection
positive_peaks, _ = find_peaks(accel_data['filtered'], height=0.5, distance=fs*0.5)  # Positive peaks
negative_peaks, _ = find_peaks(-accel_data['filtered'], height=0.5, distance=fs*0.5)  # Negative peaks
step_count_filtered = len(positive_peaks) + len(negative_peaks)

# Fourier analysis on filtered data
time = accel_data["Time (s)"].values
signal = accel_data["filtered"].values
N = len(signal)
T = time[1] - time[0]
fs = 1 / T
fourier = np.fft.fft(signal, N)
psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, T)
L = np.arange(1, int(N/2))

# Create DataFrame for PSD
chart_data_psd = pd.DataFrame(np.transpose(np.array([freq[L], psd[L].real])), columns=["freq", "psd"])

# Create DataFrame for raw and filtered acceleration data
chart_data_accel = pd.DataFrame({
    'Time (s)': accel_data['Time (s)'],
    'Raw Acceleration': accel_data['Linear Acceleration y (m/s^2)'],
    'Filtered Acceleration': accel_data['filtered']
})

# GPS data analysis
gps_data['coords'] = gps_data.apply(lambda row: (row['Latitude (°)'], row['Longitude (°)']), axis=1)
distances = [geodesic(gps_data['coords'].iloc[i], gps_data['coords'].iloc[i+1]).meters for i in range(len(gps_data)-1)]
total_distance = sum(distances)
total_time = (gps_data['Time (s)'].iloc[-1] - gps_data['Time (s)'].iloc[0])  # in seconds
average_speed = total_distance / total_time  # in meters per second

# Step length calculation
if step_count_filtered > 0:
    step_length = total_distance / step_count_filtered
else:
    step_length = 0

# Calculate step count using Fourier transform
dominant_freq = freq[L][np.argmax(psd[L].real)]
step_count_fourier = dominant_freq * total_time

# Streamlit app
st.title("Acceleration and GPS Data Analysis")

# Raw and filtered acceleration data
st.subheader("Raw and Filtered Acceleration Data (y-axis)")
st.line_chart(chart_data_accel, x='Time (s)', y=['Raw Acceleration', 'Filtered Acceleration'])

# Power spectral density
st.subheader("Power Spectral Density")
st.line_chart(chart_data_psd, x='freq', y='psd', y_label='Teho', x_label='Taajuus [Hz]')

# Route on map using Folium
st.subheader("Route on Map")
df = gps_data[['Latitude (°)', 'Longitude (°)', 'Horizontal Accuracy (m)', 'Vertical Accuracy (m)']].copy()
lat_mean = df['Latitude (°)'].mean()
long_mean = df['Longitude (°)'].mean()
mymap = folium.Map(location=[lat_mean, long_mean], zoom_start=14)

# polyline 
for i in range(len(df) - 1):
    start_coords = df.iloc[i][['Latitude (°)', 'Longitude (°)']].values
    end_coords = df.iloc[i+1][['Latitude (°)', 'Longitude (°)']].values
    folium.PolyLine([start_coords, end_coords], color='blue', weight=2.5, opacity=1).add_to(mymap)

#  map and results side by side
col1, col2 = st.columns([2, 1])

with col1:
    st_folium(mymap, width=700, height=500)

with col2:
    st.subheader("Results")
    st.write(f"Step count (filtered): {step_count_filtered}")
    st.write(f"Step count (Fourier): {step_count_fourier:.2f}")
    st.write(f"Average speed: {average_speed:.2f} m/s")
    st.write(f"Total distance: {total_distance:.2f} m")
    st.write(f"Step length: {step_length:.2f} m")

colInfo = st.columns(2)
st.subheader("About the Results")
st.write(f"Total distance roughly actually walked: ~350m")
st.write(f"My assesment is that the calculations are very accurate for the data taken.")
from PIL import Image
image = Image.open('actualtravel.png')
st.image(image, caption='Real path taken, roughly 350m', use_column_width=True)
st.write(f"Conclution is that my phone is on its last straw, this is the best it can do.")
st.write(f"My walk was brisk and I was walking at a steady pace.")
st.write(f"Therefore, I believe that the average speed, and step length are accurate!")