Housing Prices Analysis in London (1995-2024)
Project Overview
This project involves an in-depth analysis of the housing market in London, focusing on the evolution of housing prices from 1995 to 2024. The dataset used includes residential properties, featuring both historical and current price estimates, along with property details such as location, size, and more. The goal of this project is to uncover trends, identify key factors influencing property prices, and use machine learning to predict future property values.

Key Objectives
Analyze the geographical distribution of property prices.
Explore the historical evolution of prices, particularly the price per square meter.
Apply predictive models to forecast future property prices.
Provide visual insights into the housing market through various maps and charts.
Dataset
The dataset used in this project contains over 200,000 records of residential properties in London, with details on:

Address & Geographic Coordinates: Full address, postal code, and latitude/longitude.
Property Details: Bedrooms, bathrooms, floor area, living rooms, etc.
Pricing Information: Historical sale prices, rental estimates, and current price estimates.
Energy Ratings & Tenure: Property energy efficiency and ownership type (freehold, leasehold).
Transaction History: Price changes over time and historical sale prices.
Steps Taken in the Project
1. Data Cleaning
Missing Values: Removed records with missing or invalid data in essential columns like saleEstimate_currentPrice and floorAreaSqM.
Outlier Removal: Excluded outliers based on price and floor area to ensure the data’s consistency.
2. Geospatial Analysis
Mapping: Used Folium and GeoPandas to visualize the geographical distribution of property prices across London.
Price per Square Meter: Calculated the price per square meter for each property and analyzed it geographically.
3. Historical Price Evolution
Time Series Analysis: Analyzed historical trends in property prices over time, focusing on price per square meter from 1995 to 2024.
Price Distribution: Examined how prices vary across different regions and how they have evolved over the years.
4. Predictive Modeling
Model Training: Trained a regression model using features such as floor area, number of bedrooms, and geographical coordinates to predict property prices.
Evaluation: Evaluated the model's performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
Feature Importance: Identified the key factors affecting property prices based on model output.
5. Visualization & Insights
Bar Charts: Displayed average prices by geographical zones (outcodes) and regions of London.
Heatmaps: Used heatmaps to visualize price variations and trends across the city.
Scatter Plots: Created scatter plots comparing actual prices to predicted values.
Technologies Used
Python: The core language for data analysis and modeling.
Pandas: For data manipulation and cleaning.
Matplotlib/Seaborn: For visualizations.
Folium: For interactive maps and geospatial visualizations.
GeoPandas: For handling geospatial data and visualizing price distributions.
Scikit-learn: For building machine learning models and evaluating their performance.
Key Results
Price Evolution: Property prices per square meter in London have seen significant fluctuations, with particular peaks and dips tied to economic conditions, demand, and supply.
Geographical Distribution: Certain areas, like central London and prime boroughs, have consistently higher property prices, while peripheral areas show a wider range of prices.
Predictive Accuracy: The regression model demonstrated good predictive power, with MAE and RMSE showing reasonable error margins for real estate forecasting.
Feature Insights: The most important factors in predicting property prices were found to be the floor area, historical prices, and geographical location.
Future Work & Improvements
Advanced Models: Test more complex machine learning models (e.g., Random Forest, Gradient Boosting) to further improve prediction accuracy.
Market Segmentation: Segment the analysis by property types (e.g., flats, maisonettes, houses) and examine their price evolution separately.
Time Series Forecasting: Use time series forecasting models (e.g., ARIMA, Prophet) to predict future price trends.