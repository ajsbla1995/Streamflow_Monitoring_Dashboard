import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from IPython.display import Image
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
from io import StringIO
import matplotlib.pyplot as plt
import time
import pull_data_from_pythonanywhere_pg
import make_pydeck 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import altair as alt
import sshtunnel
from sshtunnel import SSHTunnelForwarder
import streamlit.components.v1 as components

# Retrieves database and ssh credentials 
load_dotenv()


# ----------------------------------------------- SET WORKING DIRECTORY --------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
os.chdir(parent_dir)


# -------------------------------------------- IMPORT CA COUNTIES/STATE .SHP -> GDF -----------------------------------
# Import the CA Counties layer
county_data_directory = os.path.join(script_dir, 'Data', 'CA_Counties')  # Subfolder for data
path_to_county_shp = os.path.join(county_data_directory, 'CA_Counties_TIGER2016.shp')
counties_gdf = gpd.read_file(path_to_county_shp).to_crs("EPSG:4236")

# Import CA State layer
state_data_directory = os.path.join(script_dir, 'Data', 'CA_State')  # Subfolder for data
path_to_state_shp = os.path.join(state_data_directory, 'CA_State_TIGER2016.shp')
state_gdf = gpd.read_file(path_to_state_shp).to_crs("EPSG:4326")

# Import USGS Image file
image_directory = os.path.join(script_dir, 'Data', 'Images')  # Subfolder for data
path_to_usgs_image = os.path.join(image_directory, 'USGS_image.png')


# Function to merge counties and sensor dataframes 
def merge_sensors_counties_gdf(sensor_gdf, counties_gdf): 
    sensors_with_counties_gdf = (gpd
        .sjoin(sensor_gdf, counties_gdf[['NAMELSAD', 'geometry']], how='left', predicate = 'within')
        .drop(columns = 'index_right')
        .rename(columns = {'NAMELSAD': 'county_name'})
        .drop_duplicates()
    )
    return sensors_with_counties_gdf


# -------------------------------------------- STREAMLIT APP --------------------------------------------

def create_dashboard_page_layout():
    # Set the page configuration
    st.set_page_config(page_title="Groundwater Dashboard", 
                       layout="wide",
                        initial_sidebar_state="expanded")

    # Add a title and description
    st.title(f"California Streamflow Monitoring Dashboard")
    st.markdown("""
    Explore California's current and historical streamflow conditions across multiple USGS monitoring stations.  
    """)
 


# ------- SELECT BOXES ------------------
def create_county_selectbox(merged_sensor_data_gdf):
    '''
     Returns value that is currently selected in the selectbox.
     '''
    
    st.header("Select County to view sensors")
    county_names = sorted(merged_sensor_data_gdf['county_name'].fillna('Unspecified County').unique())
    return st.selectbox("Choose a County", ["Show All Sensors"] + list(county_names))



def create_sensor_selectbox(merged_sensor_data_gdf, selected_county):
    """
    Returns value that is currently selected in the selectbox.
    Each option is displayed as "River Name (thing_id)" to differentiate sensors along the same river.

    Args:
        merged_sensor_data_gdf (GeoDataFrame): The GeoDataFrame containing sensor data.
        selected_county (str): The selected county name or "Show All Sensors" for all counties.

    Returns:
        str: The currently selected sensor.
    """
    if selected_county == 'Show All Sensors':
        # Filter to get unique combinations of nearest_stream_or_river and thing_id
        sensors = merged_sensor_data_gdf[['nearest_stream_or_river', 'thing_name']].drop_duplicates()
    else:
        # Filter by county and then get unique combinations
        sensors = merged_sensor_data_gdf[
            merged_sensor_data_gdf['county_name'] == selected_county
        ][['nearest_stream_or_river', 'thing_name']].drop_duplicates()

    # Drop rows where either column is null
    sensors = (sensors
               .dropna(subset=['nearest_stream_or_river', 'thing_name'])
               .sort_values(by = 'nearest_stream_or_river', ascending = True)
    )

    # Create a label column for the selectbox
    sensors['label'] = sensors.apply(
        lambda row: f"{row['nearest_stream_or_river']} ({row['thing_name']})", axis=1
    )

    # Use the label column for display in the selectbox
    selected_sensor = st.selectbox(
        "Choose a USGS Station for Current Data",
        ["Select All"] + sensors['label'].tolist()
    )
    
    return selected_sensor




# ------ FILTER DATA -------------------------
def filtered_data_by_county_gdf(merged_sensor_data_gdf, county_selectbox):
    if county_selectbox == 'Show All Sensors':
        return merged_sensor_data_gdf
    return merged_sensor_data_gdf[merged_sensor_data_gdf['county_name'] == county_selectbox]

def filtered_data_by_sensor_gdf(merged_sensor_data_gdf, sensor_selectbox):
    if sensor_selectbox == "Select All":
        return merged_sensor_data_gdf
    return merged_sensor_data_gdf[merged_sensor_data_gdf['thing_name'] == sensor_selectbox.split("(")[-1].strip(")")]

def filtered_data_by_sensor_gdf(merged_sensor_data_gdf, sensor_selectbox):
    """
    Filters the GeoDataFrame based on the selected sensor from the selectbox.

    Args:
        merged_sensor_data_gdf (GeoDataFrame): The GeoDataFrame containing sensor data.
        sensor_selectbox (str): The selected sensor label from the selectbox.

    Returns:
        GeoDataFrame: The filtered GeoDataFrame based on the selected sensor.
    """
    if sensor_selectbox == "Select All":
        # Return the entire dataset if "Select All" is selected
        return merged_sensor_data_gdf

    # Extract the thing_id from the selected label
    selected_thing_id = sensor_selectbox.split("(")[-1].strip(")")

    # Filter the GeoDataFrame by the extracted thing_id
    return merged_sensor_data_gdf[merged_sensor_data_gdf['thing_name'] == selected_thing_id]

def most_recent_usgs_gdf(merged_sensor_data_gdf):
    gdf = merged_sensor_data_gdf.sort_values('obs_time').groupby('thing_name', as_index=False).last() 
    return gdf


def selected_sensor_most_recent_usgs_data_to_graph_gdf(most_recent_usgs_gdf, selected_sensor):
    gdf= most_recent_usgs_gdf[most_recent_usgs_gdf['thing_name']==selected_sensor]
    return gdf

def selected_sensor_most_recent_usgs_data_to_graph_gdf(most_recent_usgs_gdf, selected_sensor):
    
    # Extract the thing_id from the selected label
    selected_thing_id = sensor_selectbox.split("(")[-1].strip(")")
    return most_recent_usgs_gdf[most_recent_usgs_gdf['thing_name']==selected_thing_id]
    












# --------------- CREATE PYDECK MAP -----------------
def create_pydeck_map(data, 
                      counties_gdf, 
                      state_gdf,
                      zoom_level,
                      background_style = 'DARK',  
                      center_lat=None, 
                      center_lon=None):
    print(f"Zoom level: {zoom_level}")
    return make_pydeck.make_dynamic_pydeck_map(data, counties_gdf, state_gdf, background_style, zoom_level, center_lat, center_lon)





# --------------- CREATE GRAPHS -------------------------
def deprecated_create_streamflow_graph(current_data, historical_max_data, historical_mean_data):
    fig = go.Figure()

    # Resample historical_max_data to daily frequency (one row per day)
    if 'obs_time' in historical_max_data and not historical_max_data.empty:
        # Ensure obs_time is datetime
        historical_max_data['obs_time'] = pd.to_datetime(historical_max_data['obs_time'])
        
        # Create a date-only column
        historical_max_data['obs_date'] = historical_max_data['obs_time'].dt.date
        
        # Group by date and take the first row of each group
        historical_max_data = historical_max_data.groupby('obs_date').first().reset_index()


    # Resample historical_mean_data to daily frequency (one row per day)
    #if 'obs_time' in historical_mean_data and not historical_mean_data.empty:
        # Ensure obs_time is datetime
     #   historical_mean_data['obs_time'] = pd.to_datetime(historical_mean_data['obs_time'])
        
        # Create a date-only column
      #  historical_mean_data['obs_date'] = historical_mean_data['obs_time'].dt.date
        
        # Group by date and take the first row of each group
       # historical_mean_data = historical_mean_data.groupby('obs_date').first().reset_index()

    # Current Streamflow
    fig.add_trace(go.Scatter(
        x=current_data['obs_time'],
        y=current_data['discharge_streamflow_ft3_sec'],
        mode='lines',
        name='Current Streamflow',
        line=dict(color='lightblue'),
        hovertemplate='Date of Discharge: %{customdata}<br>Current Discharge: %{y} (ft³/second)<br><extra></extra>',
        customdata=current_data['obs_time']
    ))



    # Historical Daily Max
    # Check if the 'max_va' column exists in the historical data. if it doesn't, this means the historical dataframe was empty, and we'll skip this 
    if 'max_va' in historical_max_data.columns and not historical_max_data.empty:
        fig.add_trace(go.Scatter(
            x=historical_max_data['obs_time'],
            y=historical_max_data['max_va'],
            mode='markers',
            name='Historical Max Discharge for that day (1970-2023)',
            line=dict(color='red', dash='dashdot'),
            hovertemplate='Date of Max Discharge: %{customdata[0]}<br>Max Discharge: %{y} (ft³/second)<br><extra></extra>',
            customdata=historical_max_data[['day_month_stripped']]
    ))
        
    # Historical Daily Mean
    # Check if the 'max_va' column exists in the historical data. if it doesn't, this means the historical dataframe was empty, and we'll skip this 
    if 'mean_va' in historical_mean_data.columns and not historical_mean_data.empty:
        fig.add_trace(go.Scatter(
            x=historical_mean_data['obs_time'],
            y=historical_mean_data['mean_va'],
            mode='lines',
            name='Historical Mean Discharge for that day (1970-2023)',
            line=dict(color='yellow', dash='dashdot'),
            hovertemplate='Date of Mean Discharge: %{customdata[0]}<br>Mean Discharge: %{y} (ft³/second)<br><extra></extra>',
            customdata=historical_mean_data[['day_month_stripped']]
    ))
    
     # Extract values for the Title 
    sensor_id = current_data['thing_id'].iloc[0] if 'thing_id' in current_data else "Unknown Sensor"
    nearest_river = current_data.sort_values(by='nearest_stream_or_river', ascending=False).iloc[0]['nearest_stream_or_river']
    

    fig.update_layout(
        showlegend=True,
        title=f"Current and Historical Streamflow for sensor {sensor_id} at {nearest_river}",
        xaxis=dict(
            title="Date",
            titlefont=dict(size=14),
            tickformat="%Y-%m-%d",  # Format date as Year-Month-Day
        ),
        yaxis=dict(
            title="Discharge (ft³/second)",
            titlefont=dict(size=14),
        ))
    
    return fig

# --------------- CREATE GRAPHS -------------------------
def create_streamflow_graph(current_data):
    fig = go.Figure()

    # Current Streamflow
    fig.add_trace(go.Scatter(
        x=current_data['obs_time'],
        y=current_data['discharge_streamflow_ft3_sec'],
        mode='lines',
        name='Current Streamflow',
        line=dict(color='lightblue'),
        hovertemplate='Time of Last Observation: %{customdata}<br>Current Discharge: %{y} (ft³/second)<br><extra></extra>',
        customdata=current_data['obs_time'].dt.strftime('%Y-%m-%d %H:%M %p')
    ))




    # Historical Daily Max
    # Check if the 'max_va' column exists in the historical data. if it doesn't, this means the historical dataframe was empty, and we'll skip this 
    #if 'max_va' in historical_max_data.columns and not historical_max_data.empty:
    fig.add_trace(go.Scatter(
        x=current_data['obs_time'],
        y=current_data['max_historical_streamflow_ft3_sec'],
        mode='markers',
        name='Historical Max Discharge for Time of Year (1970-2023)',
        line=dict(color='red', dash='dashdot'),
        hovertemplate='Date of Max Discharge: %{customdata}<br>Max Discharge: %{y} (ft³/second)<br><extra></extra>',
        #customdata = current_data[['day_month_max']],
        customdata = current_data['day_month_max'].dt.strftime('%Y-%m-%d')
        #customdata=current_data[['day_month_max']]
))
    
    # Historical Daily Mean
    # Check if the 'max_va' column exists in the historical data. if it doesn't, this means the historical dataframe was empty, and we'll skip this 
    #if 'mean_va' in historical_mean_data.columns and not historical_mean_data.empty:
    fig.add_trace(go.Scatter(
        x=current_data['obs_time'],
        y=current_data['50th_percentile'],
        mode='lines',
        name='Historical Median Discharge for Time of Year (1970-2023)',
        line=dict(color='yellow', dash='dashdot'),
        hovertemplate='<br>Median Discharge: %{y} (ft³/second)<br><extra></extra>',
    
))

     # Extract values for the Title 
    sensor_id = current_data['thing_name'].iloc[0] if 'thing_name' in current_data else "Unknown Sensor"
    nearest_river = current_data.sort_values(by='nearest_stream_or_river', ascending=False).iloc[0]['nearest_stream_or_river']
    

    fig.update_layout(
        showlegend=True,
        title=f"Current and Historical Streamflow for sensor {sensor_id} at {nearest_river}",
        xaxis=dict(
            title="Date",
            titlefont=dict(size=14),
            tickformat="%Y-%m-%d",  # Format date as Year-Month-Day
        ),
        yaxis=dict(
            title="Discharge (ft³/second)",
            titlefont=dict(size=14),
        ))
    
    return fig


def create_county_7_day_streamflow_graph(filtered_county_data_gdf):

    '''
    Filters dataset by the most recent observation, extracts the day of year. Calculates 7 days below this
    (day_year - 7). Selects random row where day_year is within (day_year-7) -- only one row will be selected for
    each day, for each sensor.
    
    '''
    # Filter dataset to extract day of year of most recent obs.
    recent_filtered_county_data_gdf = (filtered_county_data_gdf
                               .sort_values('obs_time')
    )
    current_day_of_yr = recent_filtered_county_data_gdf['day_of_year'].iloc[0]

    # Query all those day_of_year that are up to 7 days less than the current day of year 
    last_7_days = filtered_county_data_gdf[filtered_county_data_gdf['day_of_year'] >= (current_day_of_yr - 7)]

     # Group by county and day, then sample one row per day
    sampled_data = (last_7_days
                    .groupby(['thing_name', 'county_name', 'day_of_year'])
                    .apply(lambda group: group.sample(1))  # Randomly sample 1 row per day per county
                    .reset_index(drop=True))
    # Get all unique sensors
    unique_county_sensors = sampled_data['thing_name'].unique()

 

    fig = go.Figure()
    # Current Streamflow
    for sensor_name, sensor_data in sampled_data.groupby('thing_name'):

        # Make the lines colored, to match the pydeck map. Sort sensor_data, take most recent % difference, update dictionary
        sorted = sensor_data.sort_values('obs_time', ascending = False)
        median = sorted['discharge_streamflow_ft3_sec'].iloc[0]
        percentile_10th = sorted['10th_percentile'].iloc[0]
        percentile_25th = sorted['25th_percentile'].iloc[0]
        percentile_50th = sorted['50th_percentile'].iloc[0]
        percentile_75th = sorted['75th_percentile'].iloc[0]
        percentile_90th = sorted['90th_percentile'].iloc[0]

       
        if median < percentile_10th:
            color = 'rgba(255, 0, 0, 1)'  # Red
        elif percentile_10th <=  median < percentile_25th:
            color = 'rgba(255, 165, 0, 1)'  # Orange
        elif percentile_25th <= median < percentile_75th:
            color = 'rgba(50, 205, 50, 255)'  # Green
        elif percentile_75th <= median < percentile_90th:
            color = 'rgba(0, 100, 255, 1)'  # Electric Blue
        elif median >= percentile_90th:
            color = 'rgba(153, 102, 255, 1)'  # Deep Lavender
        else:
            color = 'rgba(255, 255, 255, 1)'  # White for NaN


          # Create customdata as a list of lists
        customdata = list(zip(
            sensor_data['obs_time'].dt.strftime('%Y-%m-%d %H:%M %p'),
            sensor_data['nearest_stream_or_river'],
            sensor_data['county_name'],
            sensor_data['percent_difference_streamflow_from_median']
    ))

       # Plot the trend for each sensor 
        fig.add_trace(go.Scatter(
            x=sensor_data['obs_time'],
            y=sensor_data['discharge_streamflow_ft3_sec'],
            mode='lines',
            name=sensor_name,
            line=dict(color=color),
            hovertemplate=(
            '<b>Sensor Name:</b> %{meta}<br>'
            '<b>Nearest Stream or River:</b> %{customdata[1]}<br>'
            '<b>Time of Observation:</b> %{customdata[0]}<br>'
            '<b>Discharge:</b> %{y} ft³/second<br><extra></extra>'
        ),
            customdata=customdata,
            meta=sensor_name  # Explicitly assign the sensor name as metadata
        ))




   
     # Extract values for the Title 
    
    fig.update_layout(
        showlegend=True,
        title=f"7 Day Streamflow Trend of All Sensors in {filtered_county_data_gdf['county_name'].iloc[0]}",
        xaxis=dict(
            title="Date",
            titlefont=dict(size=14),
            tickformat="%Y-%m-%d",  # Format date as Year-Month-Day
        ),
        yaxis=dict(
            title="Discharge (ft³/second)",
            titlefont=dict(size=14),
        ))
    
    return fig




# ---------------- CREATE BARCHARTS, DONUT GRAPHS, SUMMARY STATS / METRIC BOXES ---------------------
def deprecated_add_streamflow_single_sensor_metric(type, gdf):
    '''
    Sort the gdf by the highest % streamflow difference-- select the desired ranking. Queries the top 5 
    streamflow discharges, select any ranking in the top 5 to display on the dashboard

    ARGS
    -----
    gdf : filtered gdf you want to sort order by % streamflow difference
    ranking: Choose any of (0,1,2,3,4) for the ranked streamflows 
    '''

    gdf_sorted = gdf.sort_values('obs_time', ascending= False)
    # Top 3 Highest Discharge Change from Mean Number 

    
    #label = f"Sensor at {most_discharge['nearest_stream_or_river'].iloc[ranking]}, {most_discharge['county_name'].iloc[ranking]} "
        # Create the label with "Sensor at ..." on top and county name below
    
    if type == 'current':
        label = 'Current Streamflow (ft³/sec)'
        value = f"{gdf_sorted['discharge_streamflow_ft3_sec'].iloc[0]} ft³/sec"
        delta_sensor = f"{gdf_sorted['percent_difference_streamflow_from_mean'].iloc[0]}%"
        return st.metric(label, 
                            value, 
                            delta = delta_sensor, 
                            delta_color="normal",
                            help ='Current streamflow shown with percent difference from the mean streamflow of that sensor for time of year', 
                            label_visibility="visible")
    elif type == 'max_historical':
        #label = f'Max Historical Streamflow (ft³/sec)'
        value = f"{int(round(gdf_sorted['max_historical_streamflow_ft3_sec'].iloc[0]))} ft³/sec"
        label = 'Max Historical Streamflow (ft³/sec)'
        return st.metric(label, 
                         value, 
                         help = f'The historical max streamflow occurred on {gdf_sorted['day_month_max'].iloc[0]}',
                         label_visibility="visible",
                         )
    elif type == 'mean_historical':
        label = 'Mean Historical Streamflow (ft³/sec)'
        value = f"{int(round(gdf_sorted['mean_historical_streamflow_ft3_sec'].iloc[0]))} ft³/sec"
        return st.metric(label, 
                         value, 
                         help= "Based on the current date's average streamflow from 1970-2024", 
                         label_visibility="visible")

                       

    else:
        print(f"Invalid type: {type}. Must be 'highest' or 'lowest'.")
        return None


def add_streamflow_single_sensor_metric(type, gdf):
    '''
    Sort the gdf by the highest % streamflow difference from median-- select the desired ranking. Queries the top 5 
    streamflow discharges, select any ranking in the top 5 to display on the dashboard

    ARGS
    -----
    gdf : filtered gdf you want to sort order by % streamflow difference
    type: 'current' , 'median', 'max_historical'
    '''

    gdf_sorted = gdf.sort_values('obs_time', ascending= False)
    # Top 3 Highest Discharge Change from Mean Number 

    
    #label = f"Sensor at {most_discharge['nearest_stream_or_river'].iloc[ranking]}, {most_discharge['county_name'].iloc[ranking]} "
        # Create the label with "Sensor at ..." on top and county name below
    
    if type == 'current':
        label = 'Current Streamflow (ft³/sec)'
        value = f"{gdf_sorted['discharge_streamflow_ft3_sec'].iloc[0]} ft³/sec"
        delta_sensor = f"{gdf_sorted['percent_difference_streamflow_from_median'].iloc[0]}%"
        return st.metric(label, 
                            value, 
                            delta = delta_sensor, 
                            delta_color="normal", 
                            help ='Current streamflow shown with percent difference from the median streamflow of that sensor for time of year', 
                            label_visibility="visible")
    elif type == 'max_historical':
        #label = f'Max Historical Streamflow (ft³/sec)'
        value = f"{int(round(gdf_sorted['max_historical_streamflow_ft3_sec'].iloc[0]))} ft³/sec"
        label = 'Max Historical Streamflow (ft³/sec)'
        return st.metric(label, 
                         value, 
                         help = f'The historical max streamflow occurred on {gdf_sorted['day_month_max'].iloc[0]}',
                         label_visibility="visible",
                         )
    elif type == 'median':
        label = 'Median Historical Streamflow (ft³/sec)'
        value = f"{int(round(gdf_sorted['50th_percentile'].iloc[0]))} ft³/sec"
        return st.metric(label, 
                         value, 
                         help= "Based on the current date's median streamflow from 1970-2024", 
                         label_visibility="visible")

    else:
        print(f"Invalid type: {type}. Must be 'current', 'median' or 'max_historical'.")
        return None



def add_streamflow_ranked_metric_mean(type, gdf, ranking):
    '''
    Sort the gdf by the highest % streamflow difference-- select the desired ranking. Queries the top 5 
    streamflow discharges, select any ranking in the top 5 to display on the dashboard

    ARGS
    -----
    gdf : filtered gdf you want to sort order by % streamflow difference
    ranking: Choose any of (0,1,2,3,4) for the ranked streamflows 
    '''

    if type == 'highest':
        # Top 3 Highest Discharge Change from Mean Number 
        most_discharge = gdf.sort_values('percent_difference_streamflow_from_mean', ascending = False).head(5)
        #label = f"Sensor at {most_discharge['nearest_stream_or_river'].iloc[ranking]}, {most_discharge['county_name'].iloc[ranking]} "
         # Create the label with "Sensor at ..." on top and county name below
        
        label = (
                f"Sensor at {most_discharge['nearest_stream_or_river'].iloc[ranking]} || "
                 f"{most_discharge['county_name'].iloc[ranking]}"
        )
        
        value = f"{most_discharge['discharge_streamflow_ft3_sec'].iloc[ranking]} ft³/sec"
        delta_sensor = f"{most_discharge['percent_difference_streamflow_from_mean'].iloc[ranking]}%"
        return st.metric(label, 
                         value, 
                         delta = delta_sensor, 
                         delta_color="normal", 
                         help= "% Difference from mean historical streamflow (1970-2024) for time of year", 
                         label_visibility="visible")

    elif type == 'lowest':
        # Top 3 Highest Discharge Change from Mean Number 
        least_discharge = filtered_county_data_gdf.sort_values('percent_difference_streamflow_from_mean', ascending = True).head(5)
        label = (
                f"Sensor at {least_discharge['nearest_stream_or_river'].iloc[ranking]} || "
                 f"{least_discharge['county_name'].iloc[ranking]}"
        )
        value = f"{least_discharge['discharge_streamflow_ft3_sec'].iloc[ranking]} ft³/sec"
        delta_sensor = f"{least_discharge['percent_difference_streamflow_from_mean'].iloc[ranking]}%"
        return st.metric(label, 
                         value, 
                         delta = delta_sensor, 
                         delta_color="normal", 
                         help=" % Difference from mean historical streamflow (1970-2024) for time of year",  
                         label_visibility="visible")

    else:
        print(f"Invalid type: {type}. Must be 'highest' or 'lowest'.")
        return None



def add_streamflow_ranked_metric_median(type, gdf, ranking):
    '''
    Sort the gdf by the highest % streamflow difference-- select the desired ranking. Queries the top 5 
    streamflow discharges, select any ranking in the top 5 to display on the dashboard

    ARGS
    -----
    gdf : filtered gdf you want to sort order by % streamflow difference
    ranking: Choose any of (0,1,2,3,4) for the ranked streamflows 
    '''
    if type == 'highest':
        # Top 3 Highest Discharge Change from Mean Number 


        most_discharge = gdf.sort_values('percent_difference_streamflow_from_median', ascending = False).head(5)
        #label = f"Sensor at {most_discharge['nearest_stream_or_river'].iloc[ranking]}, {most_discharge['county_name'].iloc[ranking]} "
         # Create the label with "Sensor at ..." on top and county name below
        
        label = (
                f"Sensor at {most_discharge['nearest_stream_or_river'].iloc[ranking]} || "
                 f"{most_discharge['county_name'].iloc[ranking]}"
        )
        
        value = f"{most_discharge['discharge_streamflow_ft3_sec'].iloc[ranking]} ft³/sec"
        delta_sensor = f"{most_discharge['percent_difference_streamflow_from_median'].iloc[ranking]}%"
        return st.metric(label, 
                         value, 
                         delta = delta_sensor, 
                         delta_color="normal", 
                         help= "% Difference from median historical streamflow (1970-2024) for time of year", 
                         label_visibility="visible")

    elif type == 'lowest':
        # Top 3 Highest Discharge Change from Mean Number 
        least_discharge = filtered_county_data_gdf.sort_values('percent_difference_streamflow_from_median', ascending = True).head(5)
        label = (
                f"Sensor at {least_discharge['nearest_stream_or_river'].iloc[ranking]} || "
                 f"{least_discharge['county_name'].iloc[ranking]}"
        )
        value = f"{least_discharge['discharge_streamflow_ft3_sec'].iloc[ranking]} ft³/sec"
        delta_sensor = f"{least_discharge['percent_difference_streamflow_from_median'].iloc[ranking]}%"
        return st.metric(label, 
                         value, 
                         delta = delta_sensor, 
                         delta_color="normal", 
                         help=" % Difference from median historical streamflow (1970-2024) for time of year",  
                         label_visibility="visible")

    else:
        print(f"Invalid type: {type}. Must be 'highest' or 'lowest'.")
        return None
    

   
def make_donut(input_response, input_text, input_color):
    # Assign chart color based on the input color parameter
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    elif input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    elif input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    elif input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    elif input_color == 'yellow':
        chart_color = ['#FFD700', '#FFB300']  # Bright gold and amber shades

    # Prepare data for the donut chart
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    # Create the donut chart plot
    plot = alt.Chart(source).mark_arc(innerRadius=48).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    # Add percentage text inside the donut chart
    text = plot.mark_text(
        align='center',
        color="#29b5e8",  # Text color
        font="Lato",
        fontSize=28,
        fontWeight=500,
        fontStyle="italic"
    ).encode(text=alt.value(f'{input_response} %'))

    # Background chart for donut effect
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    return plot_bg + plot + text


# Function to create the pie charts for 'Above Mean' and 'Below Mean' based on the gdf
def deprecated_create_summary_pie_charts(type, gdf):
    ''' 
    Create pie charts that aggregate all sensors in scope to determine % of sensors below and above the historical mean
    for that day, choose either above mean or below mean values.

    ARGS
    -----
    type : 'above_mean', 'below_mean'

    RETURNS
    ------
    An altair pie chart either describing above or below mean 
    
    '''
    # Create boolean column to check if the sensor is above the mean
    gdf['above_mean'] = gdf['percent_difference_streamflow_from_mean'] > 0

    # Count sensors above and below the mean
    above_mean_count = gdf['above_mean'].sum()  # Number of sensors above mean
    below_mean_count = len(gdf) - above_mean_count  # Number of sensors below mean

    # Calculate the percentage for each category
    total_count = above_mean_count + below_mean_count
    above_mean_percentage = round((above_mean_count / total_count) * 100)
    below_mean_percentage = round((below_mean_count / total_count) * 100)

    # Choose what type of pie chart to show
    if type == 'above_mean':
        # Generate the pie charts for each category using the make_donut function
        above_mean_chart = make_donut(above_mean_percentage, 'Above Mean', 'blue')
        st.altair_chart(above_mean_chart, use_container_width=True)
    elif type == 'below_mean':
        below_mean_chart = make_donut(below_mean_percentage, 'Below Mean', 'red')
          # Display the charts in Streamlit
        st.altair_chart(below_mean_chart, use_container_width=True)
    else:
        print(f"Invalid {type}, must be one of 'above_mean' or 'below_mean'")

   

def create_summary_pie_charts(type, gdf):
    ''' 
    Create pie charts that aggregate all sensors in scope to determine % of sensors below and above the historical mean
    for that day, choose either above mean or below mean values.

    ARGS
    -----
    type : 'above_mean', 'below_mean', or 'at_mean'

    RETURNS
    ------
    An altair pie chart describing the selected streamflow category
    '''
    # Add the boolean columns for streamflow categories
    gdf = gdf.assign(
                above_normal=gdf['discharge_streamflow_ft3_sec'] > gdf['75th_percentile'],
                normal=(gdf['discharge_streamflow_ft3_sec'] >= gdf['25th_percentile']) &
                    (gdf['discharge_streamflow_ft3_sec'] <= gdf['75th_percentile']),
                below_normal=gdf['discharge_streamflow_ft3_sec'] < gdf['25th_percentile']
    )

    # Sum over the entire DataFrame to get the total count of each category
    total_counts = gdf[['above_normal', 'normal', 'below_normal']].sum()

    # Calculate the total count and percentages for each category
    total_count = total_counts.sum()
    percentages = round((total_counts / total_count) * 100)

    # Prepare the data for the pie chart
    counts_gdf = pd.DataFrame({
        'Category': total_counts.index,
        'Count': total_counts.values,
        'Percentage': percentages.values
    })

    # Choose what type of pie chart to show
    if type == 'above_normal':
        # Generate the pie chart for above mean percentage
        above_mean_chart = make_donut(counts_gdf[counts_gdf['Category'] == 'above_normal']['Percentage'].values[0], 'Above Mean', 'blue')
        st.altair_chart(above_mean_chart, use_container_width=True)

    elif type == 'normal':
        # Generate the pie chart for normal (at mean) percentage
        normal_mean_chart = make_donut(counts_gdf[counts_gdf['Category'] == 'normal']['Percentage'].values[0], 'At Mean', 'green')
        st.altair_chart(normal_mean_chart, use_container_width=True)

    elif type == 'below_normal':
        # Generate the pie chart for below mean percentage
        below_mean_chart = make_donut(counts_gdf[counts_gdf['Category'] == 'below_normal']['Percentage'].values[0], 'Below Mean', 'red')
        st.altair_chart(below_mean_chart, use_container_width=True)

    else:
        # Handle invalid type argument
        st.error(f"Invalid type: '{type}', must be one of 'above_normal', 'normal', or 'below_normal'.")



def create_single_sensor_pie_chart(gdf):
    ''' 
    Create pie charts that shows the % streamflow from the mean the sensor is currently at. No aggregation, simply reads
     the column 'percent_difference_streamflow_from_mean' and visualizes this in pie chart.

    ARGS
    -----
    type : 'above_mean', 'below_mean'

    RETURNS
    ------
    An altair pie chart either describing above or below mean 
    
    '''
   
    # 
    gdf = gdf.sort_values('obs_time', ascending = False)
    diff_mean = gdf['percent_difference_streamflow_from_mean'].iloc[0]


    # Choose what type of pie chart to show
    if diff_mean > 0:
        # Generate the pie charts for each category using the make_donut function
        above_mean_chart = make_donut(diff_mean, 'Above Mean', 'blue')
        st.altair_chart(above_mean_chart, use_container_width=True)
    elif diff_mean < 0:
        below_mean_chart = make_donut(diff_mean, 'Below Mean', 'red')
        st.altair_chart(below_mean_chart, use_container_width=True)
    else:
        at_mean_chart = make_donut(diff_mean, 'At Mean', 'yellow')
        st.altair_chart(at_mean_chart, use_container_width = True)



def deprecated_add_barchart(gdf):
     # Create boolean column to check if the sensor is above the mean


    # Group by county_name and calculate the count of above and below mean for each county
    county_counts = (gdf
                     .assign(above_mean = gdf['percent_difference_streamflow_from_mean'] > 0,                
                    )
                     .groupby('county_name')['above_mean'].agg(
                                                above_mean_count='sum',  # Count the number of True (above mean) values
                                                below_mean_count=lambda x: len(x) - x.sum()  # Count the number of False (below mean) values
                                            )
                    .assign(total_count = lambda x: (x['above_mean_count'] + x['below_mean_count']),
                            above_mean_percentage = lambda x: (x['above_mean_count']/ x['total_count']) * 100,
                            below_mean_percentage = lambda x: (x['below_mean_count']/ x['total_count']) * 100,      
                            )
                    .reset_index()
                    .sort_values('total_count', ascending = False)
    )

    # Now plot the bar chart
    # Display the bar chart with 'above_mean' and 'below_mean' counts for each county
    chart_data = (county_counts
                  .melt(id_vars=["county_name"], 
                        value_vars=["above_mean_count", "below_mean_count"],
                        var_name="Streamflow Category", 
                        value_name="Sensor Count")
    )
    # Plotting the chart
    return st.bar_chart( 
        chart_data.set_index('county_name').pivot(columns='Streamflow Category', values='Sensor Count'),
        color = ["#29b5e8" , "#E74C3C"],      # red: '#E74C3C', '#781F16'       blue : #29b5e8', '#155F7A'                           
    )


def add_barchart(gdf):
     # Create boolean column to check if the sensor is above the mean


    # Group by county_name and calculate the count of above and below mean for each county
    county_counts =(gdf
                    .assign(
                        above_normal=gdf['discharge_streamflow_ft3_sec'] > gdf['75th_percentile'],
                        normal=(gdf['discharge_streamflow_ft3_sec'] >= gdf['25th_percentile']) &
                            (gdf['discharge_streamflow_ft3_sec'] <= gdf['75th_percentile']),
                        below_normal=gdf['discharge_streamflow_ft3_sec'] < gdf['25th_percentile']
                    )
                    # Group by county_name and calculate counts for each streamflow category
                    .groupby('county_name')
                    .agg(
                        above_normal_count=('above_normal', 'sum'),  # Count of above-normal values
                        normal_count=('normal', 'sum'),              # Count of normal values
                        below_normal_count=('below_normal', 'sum')   # Count of below-normal values
                    )
                    .reset_index()
                    # Add total count and percentages for each category
                    .assign(
                        total_count=lambda x: x['above_normal_count'] + x['normal_count'] + x['below_normal_count'],
                        above_count_percentage=lambda x: (x['above_normal_count'] / x['total_count']) * 100,
                        normal_count_percentage=lambda x: (x['normal_count'] / x['total_count']) * 100,
                        below_count_percentage=lambda x: (x['below_normal_count'] / x['total_count']) * 100
                    )
                    .sort_values('total_count', ascending=False)
                )
    # Now plot the bar chart
    # Display the bar chart with 'above_mean' and 'below_mean' counts for each county
    chart_data = (county_counts
                  .melt(id_vars=["county_name"], 
                        value_vars=["above_normal_count", "normal_count", "below_normal_count"],
                        var_name="Streamflow Category", 
                        value_name="Sensor Count")
    )
    # Plotting the chart
    return st.bar_chart( 
        chart_data.set_index('county_name').pivot(columns='Streamflow Category', values='Sensor Count'),
        color = ["#29b5e8" , "#E74C3C", "#32CD32"],      # red: '#E74C3C', '#781F16'  lime green: #32CD32     blue : #29b5e8', '#155F7A'                           
    )


def create_legend_html():
    legend_html = """
    <div style="
        position: absolute;
        bottom: 10px;
        left: 10px;
        z-index: 1000;
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 0px 5px #00000050;
        font-size: 14px;
    ">
        <b> MAP LEGEND </b><br>
        <br>
        <b>California Streamflow Conditions</b><br>
        <br>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: rgb(153, 102, 255); margin-right: 5px;"></div> Much Above Normal
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: rgb(0, 100, 255); margin-right: 5px;"></div> Above Normal
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: rgb(50, 205, 50); margin-right: 5px;"></div> At Normal
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: rgb(255, 165, 0); margin-right: 5px;"></div> Below Normal
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: rgb(255, 0, 0); margin-right: 5px;"></div> Much Below Normal
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: rgb(255, 255, 255); margin-right: 5px; border: 1px solid #000;"></div> No Historical Data Available
        </div>
    </div>
    """
    return legend_html



# ------------------------------------ CREATE THE DASHBOARD -----------------------------------------------

create_dashboard_page_layout()

# ----------------- LAYOUT OF DASHBOARD ----------|
col1, col2 = st.columns([3, 1])


# --- PULL DATA FROM DB, MERGE COUNTY GDF WITH DB ---|


postgis_data_pull_gdf = pull_data_from_pythonanywhere_pg.postgis_to_gdf() # Make connection to postgres
counties_gdf = gpd.read_file(path_to_county_shp).to_crs("EPSG:4236") # Read in county shapefile to gpd
sensor_data_gdf = merge_sensors_counties_gdf(postgis_data_pull_gdf, counties_gdf) # Merge county and postgres data
# Get the most recent observation for each sensor, only render the latest observations 
latest_data_gdf = sensor_data_gdf.sort_values('obs_time').groupby('thing_name', as_index=False).last() 

# Get the most recent time for the 'Last Updated:' informatiom
def get_most_recent_timestamp(gdf):
    '''
    Finds the most recent observation from the dataframe, formats the most recent 'obs_time' column row to datetime 
    object from a Timestamp, strips the regional time conversion for UTC (-800), reformats the time as Hour: Minute: pm.

    ARGS
    ----
    gdf : geodataframe of the most recent sensor data

    RETURNS
    ------
    Formatted datetime object
    
    '''
    time_sorted_gdf = gdf.sort_values('obs_time', ascending = False)
    update_time = time_sorted_gdf['obs_time'].iloc[0]
    # Convert the timestamp to a datetime object if it's not already
    timestamp_converted = pd.to_datetime(update_time)
  # Convert to local time (removes timezone offset)
    timestamp_local = timestamp_converted.tz_localize(None)  # Removes the timezone information
    
    # Format as 'HH:MM AM/PM'
    #formatted_time = timestamp_local.strftime("%I:%M %p")
    
    # Format as 'HH:MM AM/PM MM/YYYY'
    formatted_time= timestamp_local.strftime("%I:%M %p, %m/%d/%Y ")

    return formatted_time

print(sensor_data_gdf) # Print df for any bugs



 




# ------------- SIDEBAR SELECTBOXES ---------|
with st.sidebar:
    st.image(path_to_usgs_image, width=150) # Add USGS image
    # County Selectbox
    county_selectbox = create_county_selectbox(sensor_data_gdf) # returns name of selection
    filtered_county_data_gdf = filtered_data_by_county_gdf(sensor_data_gdf, county_selectbox) # for pydeck map filtering
    
    # Sensor Selectbox
    sensor_selectbox= create_sensor_selectbox(filtered_county_data_gdf, county_selectbox) # returns name of selection -> Creek Name (USGS-12345678)
    sensor_selectbox_thing_id = sensor_selectbox.split("(")[-1].strip(")") # formats Selectbox value -> extracts 'USGS-12345678' to find and plot row data for unique sensor ID
    selected_sensor_data_gdf = sensor_data_gdf[sensor_data_gdf['thing_name']==sensor_selectbox_thing_id] # for plotly graphing current trends, requires selectbox selection to be formatted so only the 'USGS-12345678' is shown
                               
    
    unique_sensors = sensor_data_gdf['site_no'].nunique()

    # Map style Selectbox
    map_style = st.selectbox(
        "Select Map Style",
        options=["Dark", "Satellite"],
        index=0
    )
    #st.image(path_to_usgs_image, width=200) # Add USGS image
    #st.markdown('#####') # Add Blank Spacing
    #st.markdown('#####') # Add Blank Spacing

    # Add the legend to Streamlit
    legend_html = create_legend_html()
    components.html(legend_html, height=300, width=300)

   
# Determine the map style URL for Pydeck
if map_style == "Satellite":
    map_style_url = "mapbox://styles/mapbox/satellite-v9"
else:  # Dark theme
    map_style_url = "mapbox://styles/mapbox/dark-v10"

with st.expander('About the Dashboard', expanded=False):
    st.write('''
            - :orange[**Goal**]: This dashboard serves to report accurate near-real time streamflow data of United States Geological Survey (USGS) 
                                    monitoring stations. Historical averages and maximum streamflow data are compared for nearly each 
                                    sensor -- allowing the user to  quickly gain context and view streamflow trends in California.
                                    This dashboard may help indicate states of drought or flooding throughout California.
             
            - :orange[**About the Data**]: The data is extracted via the USGS SensorThings API, which updates sensor data approximately every hour.
                                             Historical data is also extracted from the USGS endpoint. In order to provide insight into
                                             above- and below- normal conditions, the historical streamflow data from 1970-2024 for each 
                                             sensor was divided into percentiles - 10th, 25th, 50th, 75th and 90th percentiles. A stream is 
                                             considered at normal levels if the current streamflow is between the 25th and 75th percentiles.
                                             Some sensors do not have any historical data, and so do not include any historical metrics. 
              
                Data: [USGS Sensor Things API](<https://labs.waterdata.usgs.gov/sta/v1.1/Things>)  ||  [Historical Sensor Data](<https://waterservices.usgs.gov/nwis/stats/>).
           
            - :orange[**How to Use**]: To view a particular sensor, select the county of interest in the left sidebar, 
                                         then select a sensor to view current and historic streamflow data.
            ''')


# ---------- SHOW PYDECK MAP OF ALL SENSORS --------|
if county_selectbox == 'Show All Sensors' and sensor_selectbox == "Select All":
    # Only show the most recent data, saves on rendering costs
    pydeck_map = create_pydeck_map(latest_data_gdf, counties_gdf, state_gdf, background_style= map_style_url, zoom_level= 4.6)
    with col1:
        st.pydeck_chart(pydeck_map)
        st.write(f"Showing all sensors. Last updated at {get_most_recent_timestamp(latest_data_gdf)}")
    
        st.markdown('###### COUNT OF SENSORS AT NORMAL, BELOW-NORMAL, AND ABOVE-NORMAL STREAMFLOW CONDITIONS PER COUNTY') # Title
        add_barchart(latest_data_gdf)

       
    with col2:
         # --------- ADD CALIFORNIA METRICS ---------------| 
        st.markdown('#### OVERVIEW FOR CA') # Title

        # PIE CHARTS 
        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>% SENSORS ABOVE NORMAL STREAMFLOW</div>",
            unsafe_allow_html=True,
            help = 'Above-normal streamflow is streamflow above 75th percentile calculated from daily sensor from 1970-2024 for time of year'
        )
        create_summary_pie_charts('above_normal', latest_data_gdf)


        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>% SENSORS AT NORMAL STREAMFLOW</div>",
            unsafe_allow_html=True,
            help = 'Normal streamflow is streamflow between 25th and 75th percentiles calculated from daily sensor from 1970-2024 for time of year'
        )
        create_summary_pie_charts('normal', latest_data_gdf)


        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>% SENSORS BELOW AVG STREAMFLOW</div>",
            unsafe_allow_html=True,
            help = 'Below-normal streamflow is below 25th percentile caluclated from daily averages from 1970-2024 for time of year'
        )
        create_summary_pie_charts('below_normal', latest_data_gdf)


       
        # INDIVIDUAL SENSOR PERFORMANCE 
        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>GREATEST STREAMFLOW % ABOVE MEDIAN</div>",
            unsafe_allow_html=True,
        )
        add_streamflow_ranked_metric_median(type = 'highest', gdf = latest_data_gdf, ranking = 0)
        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>GREATEST STREAMFLOW % BELOW MEDIAN</div>",
            unsafe_allow_html=True,
        
        )
        add_streamflow_ranked_metric_median(type = 'lowest', gdf = latest_data_gdf, ranking = 0)



# SHOW PYDECK MAP OF SENSORS IN FILTERED COUNTIES ----|
elif county_selectbox != 'Show All Sensors' and sensor_selectbox == "Select All":
    latest_county_gdf = filtered_county_data_gdf.sort_values('obs_time').groupby('thing_name', as_index=False).last() 
    pydeck_map = create_pydeck_map(latest_county_gdf, 
                                   counties_gdf, 
                                   state_gdf,  
                                   background_style= map_style_url, 
                                   zoom_level= 8)
    
    seven_day_fig = create_county_7_day_streamflow_graph(filtered_county_data_gdf)
    
    with col1:
        st.pydeck_chart(pydeck_map)
        st.write(f"Showing all sensors in {county_selectbox}")
        st.plotly_chart(seven_day_fig, use_container_width=True)

        # --------------------------Show relative streamflows as progress bar for each sensor per county


    with col2:
        # --------- ADD COUNTY METRICS ---------------| 
        #  
        st.markdown(f'#### OVERVIEW FOR {latest_county_gdf['county_name'].iloc[0].upper()}')

        #  PIE CHARTS
        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>% SENSORS ABOVE NORMAL STREAMFLOW</div>",
            unsafe_allow_html=True,
            help = 'Above normal streamflow is above the 75th percentile calculated from daily sensor from 1970-2024 for time of year'
        )
        create_summary_pie_charts('above_normal', latest_county_gdf)


        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>% SENSORS AT NORMAL STREAMFLOW</div>",
            unsafe_allow_html=True,
            help = 'Normal streamflow is within the 25th-75th percentile calculated from daily sensor data 1970-2024 for that time of year'
        )
        create_summary_pie_charts('normal', latest_county_gdf)


        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>% SENSORS BELOW NORMAL STREAMFLOW</div>",
            unsafe_allow_html=True,
            help = 'Below normal streamflow is below the 25th percentile calculated from daily sensor from 1970-2024 for time of year'
        )
        create_summary_pie_charts('below_normal', latest_county_gdf)

        st.markdown(f'###### ') # Add Space
         # INDIVIDUAL SENSOR PERFORMANCE 
        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>GREATEST STREAMFLOW % ABOVE MEDIAN</div>",
            unsafe_allow_html=True,
        )
        add_streamflow_ranked_metric_median(type = 'highest', gdf = latest_county_gdf, ranking = 0)

        st.markdown(
            "<div style='font-size:12px; font-weight:bold;'>GREATEST STREAMFLOW % BELOW MEDIAN</div>",
            unsafe_allow_html=True,
        
        )
        add_streamflow_ranked_metric_median(type = 'lowest', gdf = latest_county_gdf, ranking = 0)
       



else:

    fig = create_streamflow_graph(selected_sensor_data_gdf)
    pydeck_map = create_pydeck_map(selected_sensor_data_gdf, counties_gdf, state_gdf, 
                                   background_style= map_style_url, 
                                   center_lat=selected_sensor_data_gdf.geometry.iloc[0].y,
                                   center_lon=selected_sensor_data_gdf.geometry.iloc[0].x,
                                   zoom_level=9.5)
    with col1:
        st.pydeck_chart(pydeck_map)
        st.plotly_chart(fig, use_container_width=True)

        
    with col2:

        # Title 
        st.markdown(f'#### SENSOR {selected_sensor_data_gdf['thing_name'].iloc[0]}')
        #st.markdown(f'#### SENSOR OVERVIEW')
        #st.markdown(f'###### {selected_sensor_data_gdf['thing_name'].iloc[0]}')
        st.markdown(f'###### {selected_sensor_data_gdf['nearest_stream_or_river'].iloc[0].upper()}, {selected_sensor_data_gdf['county_name'].iloc[0].upper()}')
        
        st.markdown(f'###### ')
        st.markdown(f'###### ')
        st.markdown(f'###### ')

        #  PIE CHARTS
       

        #create_single_sensor_pie_chart(selected_sensor_data_gdf)
    
        add_streamflow_single_sensor_metric(type = 'current', gdf= selected_sensor_data_gdf)

        add_streamflow_single_sensor_metric(type = 'median', gdf= selected_sensor_data_gdf)
        add_streamflow_single_sensor_metric(type = 'max_historical', gdf= selected_sensor_data_gdf)
 



    
        

