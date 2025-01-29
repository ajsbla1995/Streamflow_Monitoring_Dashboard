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
import sshtunnel
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

load_dotenv()

def deprecated_postgis_to_gdf():
    # SSH and Postgres credentials
    ssh_host = "ssh.pythonanywhere.com"
    ssh_username = "ajsbla"
    ssh_password = "Wictyx-2bicva-nuhtek"  # Avoid hardcoding for security
    postgres_hostname = "ajsbla-4280.postgres.pythonanywhere-services.com"
    postgres_host_port = 14280
    postgres_user = "super"
    postgres_password = "postgres5"
    postgres_db = "postgres"
    

    # SQL Query
    sql_discharge_query = """
        SELECT * 
        FROM "usgs_water_sensor_data";
    """

    # SSH Timeout Settings
    sshtunnel.SSH_TIMEOUT = 30.0
    sshtunnel.TUNNEL_TIMEOUT = 30.0

    try:
        # Establish SSH tunnel
        with SSHTunnelForwarder(
            ssh_address_or_host=(ssh_host),
            ssh_username=ssh_username,
            ssh_password=ssh_password,  # Optional if using SSH keys
            remote_bind_address=(postgres_hostname, postgres_host_port),
        ) as tunnel:
            print(f"SSH Tunnel established on local port: {tunnel.local_bind_port}")
            
            # Connect to PostgreSQL through the tunnel
            conn = psycopg2.connect(
                user=postgres_user,
                password=postgres_password,
                host="127.0.0.1",
                port=tunnel.local_bind_port,  # Use tunnel's local port
                database=postgres_db,
            )

            # Read data into GeoDataFrame
 
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
                )
            )
            print("Data successfully loaded into GeoDataFrame.")
            print(gdf.head())  # Print sample data
            print(gdf.dtypes)  # Print data types
            return gdf

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@st.cache_data
def deprecated_postgis_to_gdf():
    
    # Accessing SSH and Postgres credentials through .env file (locally)
    #ssh_host = os.getenv("SSH_HOST")
    #ssh_username = os.getenv("SSH_USERNAME")
    #ssh_password = os.getenv("SSH_PASSWORD")
    #postgres_hostname = os.getenv("DB_HOST")
    #postgres_host_port = int(os.getenv("DB_PORT"))
    #postgres_user = os.getenv("DB_USERNAME")
    #postgres_password = os.getenv("DB_PASSWORD")
    #postgres_db = os.getenv("DB_NAME")

    # Accessing the secrets from Streamlit's secret management system
    ssh_host = st.secrets["postgres"]["SSH_HOST"]
    ssh_username = st.secrets["postgres"]["SSH_USERNAME"]
    ssh_password = st.secrets["postgres"]["SSH_PASSWORD"]
    postgres_hostname = st.secrets["postgres"]["DB_HOST"]
    postgres_host_port = int(st.secrets["postgres"]["DB_PORT"])
    postgres_user = st.secrets["postgres"]["DB_USERNAME"]
    postgres_password = st.secrets["postgres"]["DB_PASSWORD"]
    postgres_db = st.secrets["postgres"]["DB_NAME"]

    # SQL Query
    sql_discharge_query = """
        SELECT * 
        FROM "usgs_water_sensor_data";
    """

    # SSH Timeout Settings
    sshtunnel.SSH_TIMEOUT = 30.0
    sshtunnel.TUNNEL_TIMEOUT = 30.0

    try:
        # Establish SSH tunnel
        with SSHTunnelForwarder(
            ssh_address_or_host=(ssh_host),
            ssh_username=ssh_username,
            ssh_password=ssh_password,  # Optional if using SSH keys
            remote_bind_address=(postgres_hostname, postgres_host_port),
        ) as tunnel:
            print(f"SSH Tunnel established on local port: {tunnel.local_bind_port}")
            
            # Connect to PostgreSQL through the tunnel
            conn = psycopg2.connect(
                user=postgres_user,
                password=postgres_password,
                host="127.0.0.1",
                port=tunnel.local_bind_port,  # Use tunnel's local port
                database=postgres_db,
            )

            # Read data into GeoDataFrame
 
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
                )
            )
            print("Data successfully loaded into GeoDataFrame.")
            print(gdf.head())  # Print sample data
            print(gdf.dtypes)  # Print data types
            return gdf

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# ------------------------------------------------ CACHED VERSION ---------------------------
@st.cache_resource
def get_db_connection():
    """Establish an SSH tunnel and return a persistent database connection."""
    
    # Accessing SSH and Postgres credentials through .env file (locally)
    #ssh_host = os.getenv("SSH_HOST")
    #ssh_username = os.getenv("SSH_USERNAME")
    #ssh_password = os.getenv("SSH_PASSWORD")
    #postgres_hostname = os.getenv("DB_HOST")
    #postgres_host_port = int(os.getenv("DB_PORT"))
    #postgres_user = os.getenv("DB_USERNAME")
    #postgres_password = os.getenv("DB_PASSWORD")
    #postgres_db = os.getenv("DB_NAME")

    # Accessing the secrets from Streamlit's secret management system
    ssh_host = st.secrets["postgres"]["SSH_HOST"]
    ssh_username = st.secrets["postgres"]["SSH_USERNAME"]
    ssh_password = st.secrets["postgres"]["SSH_PASSWORD"]
    postgres_hostname = st.secrets["postgres"]["DB_HOST"]
    postgres_host_port = int(st.secrets["postgres"]["DB_PORT"])
    postgres_user = st.secrets["postgres"]["DB_USERNAME"]
    postgres_password = st.secrets["postgres"]["DB_PASSWORD"]
    postgres_db = st.secrets["postgres"]["DB_NAME"]

    # Start SSH tunnel
    tunnel = SSHTunnelForwarder(
        (ssh_host),
        ssh_username=ssh_username,
        ssh_password=ssh_password,  # Optional if using SSH keys
        remote_bind_address=(postgres_hostname, postgres_host_port),
    )
    tunnel.start()
    print(f"SSH Tunnel established on local port: {tunnel.local_bind_port}")

    # Connect to Postgres
    conn = psycopg2.connect(
        user=postgres_user,
        password=postgres_password,
        host="127.0.0.1",
        port=tunnel.local_bind_port,
        database=postgres_db,
    )
    return conn, tunnel  # Return both to manage tunnel lifecycle


@st.cache_data(ttl=1800)  # Cache results for 30 minutes
def postgis_to_gdf():
    """Fetch data from PostgreSQL and return a cached GeoDataFrame."""
    conn, _ = get_db_connection()  # Use cached connection
    sql_query = "SELECT * FROM usgs_water_sensor_data"

    gdf = (
        gpd.read_postgis(
            sql=sql_query,
            con=conn,
            geom_col='geometry'
        )
        .to_crs("EPSG:4236")
        .sort_values(by=['thing_name', 'obs_time'])
        .assign(
            discharge_streamflow_ft3_sec=lambda x: pd.to_numeric(x['discharge_streamflow_ft3_sec'], errors='coerce'),
            obs_time=lambda x: pd.to_datetime(x['obs_time'], errors='coerce').dt.tz_convert('US/Pacific')
        )
    )
    return gdf

def main():
    gdf = postgis_to_gdf()
    if gdf is not None:
        print("GeoDataFrame processing complete!")
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()

