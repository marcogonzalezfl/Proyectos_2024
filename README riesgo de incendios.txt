Fire Risk Index (FRI) Analysis and Interactive Visualization
This project calculates the Fire Risk Index (FRI) based on three spectral indices: NBR, NDVI, and NDMI. The results are visualized in an interactive map using OpenStreetMap as a base layer, where FRI values are represented as a heatmap.

Project Overview
Key Components:
Synthetic Data Generation:

Generates synthetic data for the indices NBR, NDVI, and NDMI with random values.
Coordinates are randomly distributed within an area of Paraguay.
FRI Calculation:

Calculates FRI as a weighted sum of the NBR, NDVI, and NDMI indices.
Classifies the FRI values into five categories: Very Low, Low, Medium, High, and Very High.
Interactive Map Visualization:

Creates an interactive map using Folium with OpenStreetMap as the base layer.
Displays FRI values as a heatmap on the map.
Requirements
Make sure you have the following dependencies installed:

numpy
pandas
folium