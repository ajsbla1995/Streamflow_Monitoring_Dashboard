import geopandas as gpd
import pandas as pd
import folium
from folium import Element
from shapely.geometry import Point
from IPython.display import Image
import seaborn as sns
import leafmap.foliumap as leafmap
from folium import FeatureGroup
from dotenv import load_dotenv
import pydeck as pdk
import streamlit as st
import json
import requests
import re
import os
from sqlalchemy import create_engine, Integer, Float, VARCHAR, DateTime, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2 import Geometry, WKTElement
import psycopg2
import logging 
from datetime import datetime
import pydeck
import streamlit as st
import seaborn as sns
from io import StringIO
import matplotlib.pyplot as plt

# --------------------------------------- API CALL TO USGS FOR HISTORICAL DATA -----------------------------------

def historical_daily_mean_usgs_sensor_stats_df(usgs_sensor_id, start_year, end_year):

    ''' 
    Make API call to waterservices.usgs.gov for historical stream data so that we can compare the 
    current levels to historical data for that particular sensor. The API call returns a response 
    that is in a tab-delimited format (rdb), so we must handle this appropriately to make a gdf.

    The USGS API call allows us to aggregate all of the daily, monthly, or annual data for a particular site. This
    will then be used to compare each point to its historical average, min and max to show relative levels.

    If no data is available for the USGS sensor, then a print message and/or error message is returned. None-type object
    is returned, 

    Arguments
    ----------
    usgs_sensor_id (object): 8-16 digit id number, not prefaced by 'USGS-'
    time_scale (obj): 'daily', 'monthly', 'annually' -> for example, 'daily' for 2023 will show 366 rows, with the mean value aggregated over however many times the sensor recorded data for that day 
    stat_type (obj): 'mean', 'max', 'min', 'median' -> how we want to aggregate the data 
    start_year (obj): what year to start the data collection.

    Returns
    -------
    dataframe 
    
    '''
        
    # USGS API Base URL
    usgs_base_url = 'https://waterservices.usgs.gov/nwis/stat/'


    # Query Parameters
    params = {
        "sites": usgs_sensor_id,  # Replace with your desired site number
        "statReportType": f"daily",     # Type of statistics report (e.g., daily, monthly, annual)
        "statType": f"mean",           # Statistic type (e.g., mean, median, max, min).
        "parameterCd" : "00060",      # Discharge data (example parameter), 00065 for Gauge Height
        "startDt": f"{start_year}",             # Start date (yr). The daily mean will, for each day, calculate the mean measurement from the start date till the end date
        "endDt": f"{end_year}"                # If not specified, it is the most recent year.
    }


    # Call the API
    response = requests.get(usgs_base_url, params = params)

    if response.status_code == 200:
        # Check if the response content is empty or if it doesn't contain tab-delimited data
        if response.text.strip():  # If there's any non-whitespace content or there is nothing there
            try:
                # Load the response content into a pandas DataFrame
                data = StringIO(response.text)
                df = (pd
                      .read_csv(data, sep="\t", comment="#", engine="python")
                      .drop(index = 0)
                      .assign(
                          month_nu=lambda x: x['month_nu'].astype(int),
                          day_nu=lambda x: x['day_nu'].astype(int),
                          count_nu=lambda x: x['count_nu'].astype(int),
                          mean_va=lambda x: x['mean_va'].astype(float).astype(int),
                          # Combine month and day into a proper date
                          day_month=lambda x: pd.to_datetime(
                              x['begin_yr'] + "-" + x['month_nu'].astype(str) + "-" + x['day_nu'].astype(str),
                              errors="coerce"),
                          
                      )
                    .assign(
                          # Now calculate day_of_year after creating the day_month column
                          day_of_year=lambda x: x['day_month'].dt.dayofyear  # Extract day of the year
                    )
                      
                )               
                # Check if the DataFrame has data
                if df.empty:
                    print("No data found for the site.")
                    return None
                return df
            except pd.errors.EmptyDataError:
                # Handle the case where the data is not in the expected format
                print(f"Error: No data could be parsed. Did not return a dataframe with streamflow data for site USGS-{usgs_sensor_id}.")
                return None
        else:
            print(f"No data for site {usgs_sensor_id}.")
            return None
    else:
        print(f"API response returned error: {response.status_code}")
        print("Response text:", response.text)  # Debugging information
        return None


def historical_daily_max_usgs_sensor_stats_df(usgs_sensor_id, start_year, end_year):
    # USGS API Base URL
    usgs_base_url = 'https://waterservices.usgs.gov/nwis/stat/'

    # Query Parameters
    params = {
        "sites": usgs_sensor_id,  # Replace with your desired site number
        "statReportType": f"daily",  # Type of statistics report (e.g., daily, monthly, annual)
        "statType": f"max",  # Statistic type (e.g., mean, median, max, min).
        "parameterCd": "00060",  # Discharge data (example parameter), 00065 for Gauge Height
        "startDt": f"{start_year}",  # Start date (yr).
        "endDt": f"{end_year}"  # End date (yr).
    }

    # Call the API
    response = requests.get(usgs_base_url, params=params)

    if response.status_code == 200:
        # Check if the response content is empty or if it doesn't contain tab-delimited data
        if response.text.strip():  # If there's any non-whitespace content or there is nothing there
            try:
                # Load the response content into a pandas DataFrame
                data = StringIO(response.text)
                df = (pd
                      .read_csv(data, sep="\t", comment="#", engine="python")
                      .drop(index=0)
                      .assign(
                          month_nu=lambda x: x['month_nu'].astype(int),
                          day_nu=lambda x: x['day_nu'].astype(int),
                          count_nu=lambda x: x['count_nu'].astype(int),
                          max_va=lambda x: x['max_va'].astype(float).astype(int),
                          # Combine month and day into a proper date
                          day_month=lambda x: pd.to_datetime(
                              x['max_va_yr'] + "-" + x['month_nu'].astype(str) + "-" + x['day_nu'].astype(str),
                              errors="coerce"),
                      )
                      .assign(
                          # Now calculate day_of_year after creating the day_month column
                          day_of_year=lambda x: x['day_month'].dt.dayofyear  # Extract day of the year
                      )
                )

                # Check if the DataFrame has data
                if df.empty:
                    print("No data found for the site.")
                    return None
                return df
            except pd.errors.EmptyDataError:
                # Handle the case where the data is not in the expected format
                print(f"Error: No data could be parsed. Did not return a dataframe with streamflow data for site USGS-{usgs_sensor_id}.")
                return None
        else:
            print(f"No data for site {usgs_sensor_id}.")
            return None
    else:
        print(f"API response returned error: {response.status_code}")
        print("Response text:", response.text)  # Debugging information
        return None