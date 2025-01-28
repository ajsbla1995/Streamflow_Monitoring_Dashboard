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
import get_historical_usgs_sensor_data



#base_dir = '/Users/andresabala/Downloads/Data Analysis Projects/Ground_Water_Dashboard/Scripts'
#os.chdir(base_dir)

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
os.chdir(parent_directory)


# ------------------------- LOGGING -------------------------------

# Configure logging
log_file = os.path.join(parent_directory, 'inserting_data_to_db.lpwdog')  # Adjust the path
logging.basicConfig(
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append to log file
        logging.StreamHandler()  # Also log to console
    ]
)

# Example log message
logging.info("------------------------||Script started||------------------------------.")


# ------------------------------- MAKE API CALL AND RETURN RESPONSE FROM USGS ---------------------
def get_groundwater_api_json_response(state_name):
    
    '''
    Makes API call to labs.waterdata.usgs.gov/sta/v1.1/ for sensor data.

    Definitions of Types of Data we can retreive 
    ---------------------------------------------
            Thing: An object of interest that has properties and is associated with one or more Datastreams.
            Datastream: A collection of observations associated with a sensor. It describes what is being observed 
                (e.g., air temperature, water quality) and links to the sensor (Thing) and the observed property.
            Observation: An individual data point or measurement recorded by a sensor. Each Observation is part of a Datastream.
            ObservedProperty: The phenomenon being observed, such as temperature, humidity, or water level.
            Sensor: The instrument or device that generates observations.

    Args
    ----
    state_name(str): State names are capitalized (i.e. 'California')
    location_type(str): The location of the sensor, i.e. ['Stream', 'Lake, Reservoir, Impoundment', 'Canal', 'Diversion',
                                                        'Ditch', 'Estuary', 'Well', 'Wetland', 'Atmosphere',
                                                        'Multiple wells', 'Extensometer well', 'Spring']

    Returns
    ------
    Census dataframe from census json response data. Creates new column ID that concats tract + block group codes, used to join TIGER boundary df.
    
    '''
   
    base_url = f"https://labs.waterdata.usgs.gov/sta/v1.1/Things"  # Water Sensor Data from USGS, this returns the metadata and URLS to the actual observations only
   
       
    
    # Build query parameters, the most recent observation is printed (top 1 results)
    params = {
        "$select": "name,properties,description,@iot.id",
        "$filter": f"properties/state eq '{state_name}'",  # Query sensors by state
       #"$resultFormat": "GeoJSON",
        "$expand": """
            Locations($select=name,description,properties,location,@iot.id),
            Datastreams(
                $select=name,description,@iot.id;
                $expand=Sensor($select=name,description,@iot.id),
                        ObservedProperty($select=name,description,@iot.id),
                        Observations(
                            $select=result,phenomenonTime,parameters,resultQuality,@iot.id;
                            $orderby=phenomenonTime desc;
                            $top=1 
                        )
            )
        """
    }
    
    headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
          }
    
    logging.info(f"Fetching data for state: {state_name}")

    try:
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code== 200:

            #print('Successful API response! Retrieving json response...')
            logging.info('Successful API response! Retrieving json response...')
            return response.json()
        else:
            #print(f"API returned error code: {response.status_code}")
            logging.warning(f"API returned error code: {response.status_code}")
            return None

            

    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeouts, etc.
        #print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
        return None


# ------------------------- NORMALIZE JSON RESPONSE TO GDF WITH APPROPRIATE DTYPE FORMATS -------------------------

def json_to_gdf(json_response): 
    """
    Converts USGS SensorThings API JSON response into a GeoPandas GeoDataFrame with appropriate dtypes.

    Args:
    -----
    json_response (dict): The JSON response from the USGS API.

    Returns:
    --------
    GeoDataFrame: A GeoPandas GeoDataFrame with sensor metadata and locations.
    """

    data = []
    
    # Iterate through the "Things" in the response
    for thing in json_response.get('value', []):
        thing_id = thing.get('@iot.id')
        thing_name = thing.get('name')
        thing_description = thing.get('description')
        thing_properties = thing.get('properties', {})
        
        # Process Locations
        for location in thing.get('Locations', []):
            loc_name = location.get('name')
            loc_description = location.get('description')
            loc_properties = location.get('properties', {})
            loc_coords = location.get('location', {}).get('coordinates', None)
            
            if loc_coords:  # Only include if coordinates exist
                # Process Datastreams
                for datastream in thing.get('Datastreams', []):
                    ds_name = datastream.get('name')
                    ds_description = datastream.get('description')
                    ds_id = datastream.get('@iot.id')
                    
                    # ObservedProperty
                    observed_property = datastream.get('ObservedProperty', {})
                    op_name = observed_property.get('name')
                    op_description = observed_property.get('description')
                    
                    # Observations
                    for observation in datastream.get('Observations', []):
                        obs_id = observation.get('@iot.id')
                        obs_result = observation.get('result')
                        obs_time = observation.get('phenomenonTime')
                        
                        # Add a row to the data list
                        data.append({
                            "thing_id": thing_id,
                            "thing_name": thing_name,
                            "thing_description": thing_description,
                            "loc_name": loc_name,
                            "loc_description": loc_description,
                            "ds_name": ds_name,
                            "ds_description": ds_description,
                            "op_name": op_name,
                            "op_description": op_description,
                            "obs_id": op_description,
                            "obs_result": obs_result,
                            "obs_time": obs_time,
                            "geometry": Point(loc_coords)
                        })
    
    # Create a GeoDataFrame, change dtypes of obs_result and obs_time to be integer and datetime, respectively.
    gdf = (gpd
           .GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")  # Assuming WGS84
          .assign(obs_result=lambda x: pd.to_numeric(x['obs_result'], errors='coerce').fillna(0).astype('int')) 
          .assign(obs_time = lambda x: pd.to_datetime(x['obs_time'], errors= 'coerce')
                   #.dt.tz_localize('UTC')  # Localize to UTC first (already in UTC from USGS)
                   .dt.tz_convert('US/Pacific')  # Convert to PST (or PDT depending on time of year)
                   .fillna(pd.NaT), # Fill NaT for any failed conversions.fillna(0))
                  #site_no=lambda x: x['thing_id']      # Use this column to merge historical on current data for mean values
                   #                 .str.replace(r'USGS-', '', regex=True),
          )
    )
    return gdf


# --------------------------- GEOLOCATE CLOSEST RIVERS TO USGS SENSORS --------------------

def geolocate_usgs_sensors(sensor_gdf):

    ''' 
    Returns geolocated sensor_gdf
    
    '''
    # Import GeoJSON containing rivers names and locations
    geojson_path = '/Users/andresabala/Downloads/Data Analysis Projects/Ground_Water_Dashboard/Scripts/Data/NHD_Named_CA_Rivers.geojson'
    NHD_rivers_gpd = (
        gpd.read_file(geojson_path)
        .to_crs(sensor_gdf.crs)  # Ensure CRS matches
    )

    # Step 1: Perform spatial join to match sensors with intersecting rivers/streams
    geolocated_rivers_gdf = gpd.sjoin(
        sensor_gdf,
        NHD_rivers_gpd[['gnis_name', 'geometry']],
        how="left",
        op="intersects"
    )

    # Step 2: Identify sensors that don't intersect any river/stream (i.e., 'gnis_name' is NaN)
    no_intersection_sensors = geolocated_rivers_gdf[geolocated_rivers_gdf['gnis_name'].isnull()]

    if not no_intersection_sensors.empty:
        # Step 3: For sensors with no intersection, find the nearest river/stream
        nearest_rivers = []

        for _, sensor in no_intersection_sensors.iterrows():
            # Calculate distances from the sensor to all rivers
            distances = NHD_rivers_gpd.geometry.distance(sensor.geometry)
            # Get the name of the nearest river
            nearest_index = distances.idxmin()
            nearest_river_name = NHD_rivers_gpd.loc[nearest_index, 'gnis_name']
            nearest_rivers.append((sensor.name, nearest_river_name))

        # Update the 'gnis_name' column for no-intersection sensors with nearest river names
        for sensor_index, river_name in nearest_rivers:
            geolocated_rivers_gdf.loc[sensor_index, 'gnis_name'] = river_name

    # Step 4: Rename the 'gnis_name' column to 'nearest_stream_or_river'
    geolocated_rivers_gdf = (geolocated_rivers_gdf
                             .rename(columns={'gnis_name': 'nearest_stream_or_river'})
                             .drop(columns = ['index_right'])
    )
    print(f"{no_intersection_sensors}")
    return geolocated_rivers_gdf
 





def add_mean_streamflow_for_time_of_year(geolocated_sensor_gdf):
    """
    Fetch historical data for all sensors in the GeoDataFrame.
    
    Parameters
    ----------
    sensor_data_gdf : GeoDataFrame
        GeoDataFrame containing sensor data with 'site_id' column.
    start_year : str
        Start year for historical data.
    end_year : str
        End year for historical data.
    
    Returns
    -------
    GeoDataFrame
        Merged GeoDataFrame with historical data for all sensors.
    """
    historical_data_list = []
    
    for index, row in geolocated_sensor_gdf.iterrows():
        usgs_sensor_id = row['simple_site_no']  # Assuming the site ID is in a column named 'site_id'
        print(f"Fetching historical data for sensor {usgs_sensor_id}...")
        
        historical_data = get_historical_usgs_sensor_data.historical_daily_mean_usgs_sensor_stats_df(
            usgs_sensor_id, 
            start_year= '1970', 
            end_year= '2023'
        )
        
        if historical_data is not None:
            # Add sensor ID for tracking
            historical_data['site_no'] = usgs_sensor_id
            historical_data_list.append(historical_data)
    
    # Combine all historical data into a single DataFrame
    if historical_data_list:
        all_historical_df = pd.concat(historical_data_list, ignore_index=True)
    else:
        print("No historical data fetched.")
        return None
    

def add_historical_mean_to_gdf(geolocated_sensor_gdf):
    """
    Adds the historical mean streamflow for the day to the sensor GeoDataFrame.

    Parameters
    ----------
    geolocated_sensor_gdf : GeoDataFrame
        The GeoDataFrame containing sensor data.

    Returns
    -------
    GeoDataFrame
        The updated GeoDataFrame with historical mean streamflow added.
    """
    historical_data_list = []

    for index, row in geolocated_sensor_gdf.iterrows():
        usgs_sensor_id = row['simple_site_no']
        obs_date = row['obs_time'].strftime('%m-%d') if pd.notnull(row['obs_time']) else None
        
        if not obs_date:
            continue  # Skip if observation date is missing
        
        logging.info(f"Fetching historical data for sensor {usgs_sensor_id}, date: {obs_date}...")
        historical_data = get_historical_usgs_sensor_data.historical_daily_mean_usgs_sensor_stats_df(
            usgs_sensor_id, 
            start_year='1970', 
            end_year='2023'
        )

        if historical_data is not None:
            # Extract the mean streamflow for the observation date
            mean_for_date = historical_data.loc[historical_data['date'] == obs_date, 'mean_streamflow']
            historical_mean = mean_for_date.values[0] if not mean_for_date.empty else None
            row['historical_mean_streamflow'] = historical_mean
            historical_data_list.append(row)

    updated_gdf = (gpd
                   .GeoDataFrame(historical_data_list, crs=geolocated_sensor_gdf.crs)
                 
                   
    )
    return updated_gdf


    

# ------------------------------------ APPEND NEW GDF TO POSTGIS DB --------------------------
def append_new_sensor_data_to_postgis(gdf):

    """
    Appends new USGS sensor data to the PostGIS-enabled PostgreSQL database.
    Only appends rows where (ds_name, obs_time) combinations do not already exist in the database. Makes connection
    to database, SELECT every ds_name and obs_time values from the database, compares the new geodataframe with existing
    database combinations. Only selects those rows that have unique ds_name/obs_time combinations. Appends unique gdf data
    to database. 
    """
    # Server credentials
    db_username = 'andresabala'
    db_password = ''
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'USGS_Water'

    # Create the engine with GeoAlchemy2 support
    engine = create_engine(
        f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    )


    logging.info("Appending new sensor data to PostGIS database.")
    
    try:
        # Query existing data to get `ds_name` and `obs_time` combinations
        with engine.connect() as conn:
            result = conn.execute(
                text('SELECT ds_name, obs_time FROM "USGS_Water_Sensor_Data"')
            )
            # Access tuple elements by index
            existing_data = set((row[0], row[1]) for row in result)

        # Filter out rows from the GeoDataFrame that already exist in the database
        new_data = gdf[~gdf.apply(
            lambda row: (row['ds_name'], row['obs_time']) in existing_data, axis=1
        )]

        # Append the filtered data to the database
        if not new_data.empty:

            # Ensure the geometry column is valid
            new_data['geometry'] = new_data['geometry'].apply(lambda geom: geom if geom is not None else None)
            new_data.to_postgis(
                'USGS_Water_Sensor_Data',  # Table name
                engine,  # SQLAlchemy engine
                if_exists='append',  # Append new data
                index=False,  # Don't write the index
                dtype={'geometry': Geometry('POINT', srid=4326)}  # Define the geometry column type
            )
            logging.info(f"{len(new_data)} new rows to append to the database!")
            #print(f"{len(new_data)} new rows successfully written to the 'USGS_Water_Sensor_Data' table.")
        else:
            #print("No new rows to add. All data already exists in the database.")
            logging.info("No new rows to add. All data already exists in the database.")

    except SQLAlchemyError as e:
        print(f"An error occurred while writing to the database: {e}")
        logging.error(f"An error occurred while writing to the database: {e}")





def main():
    """
    Main function to fetch, process, and save groundwater data.
    """

    logging.info("Main function started.")
    try:
        state_name = "California"
        response = get_groundwater_api_json_response(state_name)
        if response:
            logging.info("Converting JSON response to GeoDataFrame and Geolocating Streams.")
            ca_water_gdf = json_to_gdf(response)
            geolocated_ca_water_gdf = geolocate_usgs_sensors(ca_water_gdf)
            logging.info("Appending GeoDataFrame to PostGIS database.")
            append_new_sensor_data_to_postgis(geolocated_ca_water_gdf)
            print(append_new_sensor_data_to_postgis(geolocated_ca_water_gdf))
            #append_new_sensor_data_to_postgis(ca_water_gdf)
        logging.info("Main function completed successfully.")


    except Exception as e:
        logging.error(f"An error occurred in main(): {e}")





if __name__ == "__main__":
    main()

