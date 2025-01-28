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
import sshtunnel



def postgis_to_gdf():
    """ 
    Connects to our POSTGres Database using localhost credentials. Reads in the whole database as a geopandas dataframe.
    """
    # Database connection parameters
    db_config = {
       "dbname": "USGS_Water",
        "user": "andresabala",
        "password": "",
        "host": "127.0.0.1",  # Change if hosted on a remote server
        "port": "5432"        # Default PostgreSQL port
    }

  
   
    sql_discharge_query = """ 
        SELECT * 
        FROM "usgs_water_sensor_data";
        """
    
    try:
        # Read data into a GeoDataFrame
        conn = psycopg2.connect(**db_config)
        gdf = (gpd
               .read_postgis(
                   sql=sql_discharge_query,
                   con=conn,
                   geom_col='geometry'  # Adjust to match your geometry column
               )
               .to_crs("EPSG:4236")
               .sort_values(by=['thing_name', 'obs_time'])
               .assign(
                   discharge_streamflow_ft3_sec=lambda x: pd.to_numeric(x['discharge_streamflow_ft3_sec'], errors='coerce'),
                   # Ensure obs_time is in datetime format
                   obs_time=lambda x: pd.to_datetime(x['obs_time'], errors='coerce')
                                   .dt.tz_convert('US/Pacific'),  # Convert to Pacific Time
                    day_month_mean = lambda x: pd.to_datetime(x['day_month_mean'], errors = 'coerce'),
                    day_month_max = lambda x: pd.to_datetime(x['day_month_max'], errors = 'coerce')
                   #site_no=lambda x: x['thing_name']
                    #                .str.replace(r'USGS-', '', regex=True),
                   #percent_change_streamflow=lambda x: (x
                    #                                    .groupby('thing_id')['obs_result']
                     #                                   .pct_change() * 100
                      #                                  ).fillna(0).round(0),
                   #day_month=lambda x: x['obs_time'].dt.strftime('%Y-%m-%d'),  # Extract month-day as a string
                   #day_of_year=lambda x: x['obs_time'].dt.dayofyear
               )
        )
        
        # Convert datetime columns to string before returning
        #datetime_cols = gdf.select_dtypes(include=['datetime64[ns, US/Pacific]']).columns
        #gdf[datetime_cols] = gdf[datetime_cols].astype(str)
        print(gdf)
        print(gdf['obs_time'])
        print(gdf.dtypes)
    finally:
        conn.close()
    #print(gdf)
    return gdf


def main():
    gdf = postgis_to_gdf()
    return gdf

if __name__ == '__main__':
    main()

    



