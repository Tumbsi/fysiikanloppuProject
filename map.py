import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

gps = pd.read_csv("Location.csv")


points = [
  [gps["Latitude (°)"][i], gps["Longitude (°)"][i]] for i in range(len(gps))
]
m = folium.Map(location=[gps["Latitude (°)"].mean(), gps["Longitude (°)"].mean()], zoom_start=15)
folium.PolyLine(locations=points).add_to(m)

st_map = st_folium(m, width=900)