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
import math

# Set the working directory to the parent directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
os.chdir(parent_dir)




def make_dynamic_pydeck_map(sensors_with_counties_gdf, counties_gdf, state_gdf, map_style, zoom_level, center_lat=None, center_lon=None):

    '''
    Create a dynamic map from the postgres sensor data as a geodataframe. This function takes these geodataframes
    and converts them to json and loads this data as a geojson dictionary to be read by pydeck. 3 layers are shown: the
    sensor locations, county shapes and the state shape. The view state is changed depending on what is selected in 
    the sensorbox of the streamlit application

    ARGS
    -----
    sensors_with_counties_gdf : gdf from the postgres database containing up to date sensor information
    counties_gdf: the county shapefile converted to a gdf
    state_gdf : state shapefile converted to a gdf
    map_style: Accepts 'Satellite' and 'Dark' options
    zoom_level= dynamically changes based on the user input
    center_lat = centers map dynamically based on zoom-level

    RETURNS
    -------
    pdk.Deck object -> to enable Deck object to be shown in streamlit :  st.pydeck_chart(pydeck_map)
    
    
    '''

    # We need to serialize the datetime columns into a string. json.loads can't properly handle datetime dtypes out of the box.
    def convert_timestamps(val):
        if isinstance(val, pd.Timestamp):
            return val.isoformat()  # Convert Timestamps to string
        return val

    # Apply the conversion to the entire GeoDataFrame
    sensors_with_counties_gdf = sensors_with_counties_gdf.applymap(convert_timestamps)

    
    
    # Convert the GeoDataFrame to a GeoJSON dictionary
    usgs_sensor_geojson_dict = json.loads(sensors_with_counties_gdf.to_json())  # Pydeck requires dictionary, so we convert the json string to dictionary
    county_shp_geojson_dict = json.loads(counties_gdf.to_json())  # Pydeck requires dictionary, so we convert the json string to dictionary
    state_shp_geojson_dict = json.loads(state_gdf.to_json())

            
    
    #for feature in usgs_sensor_geojson_dict['features']:
        # Get the percent_difference_streamflow_from_mean value
    #    streamflow = feature['properties'].get('percent_difference_streamflow_from_mean', None)
    #      # Check if the value is null or NaN
    #    if change is None or math.isnan(change):
    #        feature['properties']['color'] = [255, 255, 255, 255]  # White for null or NaN values
     #   elif -100 >= change:
    #        feature['properties']['color'] = [255, 0, 0, 255]  # Red
    ##    elif -99 <= change <= -50:
    #        feature['properties']['color'] = [255, 165, 0, 255]  # Orange
     #   elif -49 <= change <= 0:
     #       feature['properties']['color'] = [255, 255, 0, 255]  # Yellow
     #   elif 0 < change <= 49:
     #       feature['properties']['color'] = [173, 216, 245, 255]  # Light Blue
     #   elif 50 <= change <= 100:
     #       feature['properties']['color'] = [0, 100, 255, 255]  # Electric Blue
     #   elif change > 100:
     #       feature['properties']['color'] = [153, 102, 255, 255]  # Deep Lavender
       
    for feature in usgs_sensor_geojson_dict['features']:
        # Get the percent_difference_streamflow_from_mean value
        streamflow = feature['properties'].get('discharge_streamflow_ft3_sec', None)
        percentile_10th = feature['properties'].get('10th_percentile', None)
        percentile_25th = feature['properties'].get('25th_percentile', None)
        percentile_50th = feature['properties'].get('50th_percentile', None)
        percentile_75th = feature['properties'].get('75th_percentile', None)
        percentile_90th = feature['properties'].get('90th_percentile', None)

        if (
            streamflow is None 
            or math.isnan(streamflow) 
            or percentile_10th is None 
            or percentile_25th is None 
            or percentile_50th is None 
            or percentile_75th is None 
            or percentile_90th is None
            ):
            feature['properties']['color'] = [255, 255, 255, 255]  # White for null or NaN values
        elif percentile_10th > streamflow:
            feature['properties']['color'] = [255, 0, 0, 255]  # Red , Much Below Normal
        elif percentile_10th <= streamflow < percentile_25th:
            feature['properties']['color'] = [255, 165, 0, 255]  # Orange , Below Normal
        elif percentile_25th <= streamflow < percentile_75th:
            feature['properties']['color'] = [50, 205, 50, 255]  # Bright Forest Green At Normal 
        elif percentile_75th <= streamflow < percentile_90th:
            feature['properties']['color'] = [0, 100, 255, 255]  # Electric Blue , Above Normal
        elif streamflow > percentile_90th:
            feature['properties']['color'] = [153, 102, 255, 255]  # Deep Lavender, Much Above Normal

    # Adjust point size based on zoom level
    point_size = max(20000 / (zoom_level + 1), 10)  # Prevent the size from being too small


    # Create a GeoJsonLayer for Pydeck using the GeoJSON dictionary
    usgs_sensor_geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        usgs_sensor_geojson_dict,  # Pass the GeoJSON dictionary
        pickable=True,
        stroked=False,
        filled=True,
        extruded=False,
    
        get_fill_color= "properties.color",
        get_radius=point_size, 
        get_line_color="[0, 0, 255, 160]",  # Blue line color (RGBA format)
    )


 
   
    # Create GeoJSONLayer of counties shapefile
    counties_shape_geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        county_shp_geojson_dict, # Pass the GeoJSON dictionary
        pickable= False,
        stroked=False,
        filled=False,
        extruded=False,
        get_fill_color="[0, 0, 0, 50]",  # Transparent fill color (RGBA format)
        get_line_color="[255, 255, 255, 255]",  # White line color (RGBA format)
    )

    # Create GeoJSONLayer of counties shapefile
    state_shape_geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        state_shp_geojson_dict, # Pass the GeoJSON dictionary
        pickable= False,
        stroked=True,
        filled=False,
        extruded=False,
        get_fill_color="[255, 255, 255, 200]",  # Transparent fill color (RGBA format)
        get_line_color="[255, 255, 255, 255]",  # White color for the polygon outline
        get_line_width = 1600,
    )

 

    # Check for NULL
    for feature in usgs_sensor_geojson_dict['features']:
    # Get the value of percent_difference_streamflow_from_mean and ensure it's valid
        change = feature['properties'].get('percent_difference_streamflow_from_mean', None)
    
    if change is None or math.isnan(change):  # Check if it's None or NaN
        feature['properties']['elevation'] = 0  # Or set to some other default value
    else:
        feature['properties']['elevation'] = change * 10  # Use the value for calculation



    center_lat = sum([feature['geometry']['coordinates'][1] for feature in usgs_sensor_geojson_dict['features']]) / len(usgs_sensor_geojson_dict['features'])
    center_lon = sum([feature['geometry']['coordinates'][0] for feature in usgs_sensor_geojson_dict['features']]) / len(usgs_sensor_geojson_dict['features'])


    view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom_level,  # Adjust zoom level based on your data's extent
            pitch=0,
            bearing=0
        )


    # Create the deck with the GeoJsonLayer
    deck = pdk.Deck(
        layers=[state_shape_geojson_layer, counties_shape_geojson_layer, usgs_sensor_geojson_layer],
        initial_view_state=view_state,
        tooltip={
            "html": """
                <b>Sensor {thing_name} at {nearest_stream_or_river}</b> <br>
                <b<County Name: </b> {county_name} <br>
                <b>Streamflow Discharge: </b> {discharge_streamflow_ft3_sec} ft³/second <br>
                <b>% Change Discharge from Historical Median: </b> {percent_difference_streamflow_from_median}%<br>
                <b>Most Recent Measurement: </b> <br>
                {obs_time}


            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "fontSize": "12px",
                "padding": "5px"
            },
        },
        map_provider="mapbox", 
        map_style= map_style # defaults to DARK style if invalid
    )
   
       
    
    
    return deck







