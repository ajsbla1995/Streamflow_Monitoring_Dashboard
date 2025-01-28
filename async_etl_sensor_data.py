import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import Point
from IPython.display import Image
import seaborn as sns
from dotenv import load_dotenv
import pydeck as pdk
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
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import aiohttp
import pandas as pd
from io import StringIO
import asyncio
import time
from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------ SET THE DIRECTORY ------------------------------------------
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
os.chdir(parent_directory)


# ------------------------------------------- MAKE API CALL TO USGS SENSORTHINGS API ------------------------------
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

    #logging.info(f"Fetching data for state: {state_name}")

    try:
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code== 200:

            #print('Successful API response! Retrieving json response...')
            #logging.info('Successful API response! Retrieving json response...')
            return response.json()
        else:
            #print(f"API returned error code: {response.status_code}")
            #logging.warning(f"API returned error code: {response.status_code}")
            return None



    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeouts, etc.
        #print(f"An error occurred: {e}")
        #logging.error(f"An error occurred: {e}")
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
                  site_no=lambda x: x['thing_id']      # Use this column to merge historical on current data for mean values
                                    .str.replace(r'USGS-', '', regex=True),
          )
                  # Now calculate day_of_year after creating the day_month column
           .assign(day_of_year=lambda x: x['obs_time'].dt.dayofyear  # Extract day of the year
            )
    )
    return gdf




# ---------------------------------------- FETCH HISTORICAL DATA USING ASYNCHRONOUS REQUESTS -----------------

async def fetch_historical_data(type, time_period, usgs_sensor_id, stat_type, start_year, end_year):
    """
    Asynchronous function to fetch historical USGS sensor data using aiohttp and NOT requests. aiohttp is an asynchronous
    version of the requests library. This allows multiple data fetching operations at once. Normally, the program will
    make a request for data, wait, then make another, etc. This skips that and allows the operations to happen concurrently.

    """
    usgs_base_url = 'https://waterservices.usgs.gov/nwis/stat/'

    if type == 'streamflow':
        parameter = "00060"
    elif type == 'gage_height':
        parameter = "00065"
    else:
        print(f"Invalid type: {type}. Must be 'streamflow' or 'gage_height'.")
        return None

    # Query Parameters
    params = {
        "sites": usgs_sensor_id,
        "statReportType": time_period,
        "statType": stat_type,
        "parameterCd": parameter,
        "startDt": f"{start_year}",
        "endDt": f"{end_year}"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(usgs_base_url, params=params) as response:
            if response.status == 200:
                response_text = await response.text()
                if response_text.strip():  # Check if content is not empty
                    try:
                        data = StringIO(response_text)
                        df = pd.read_csv(data, sep="\t", comment="#", engine="python").drop(index=0)
                        if stat_type == 'mean':
                            df = df.assign(
                                month_nu=lambda x: x['month_nu'].astype(int),
                                day_nu=lambda x: x['day_nu'].astype(int),
                                count_nu=lambda x: x['count_nu'].astype(int),
                                mean_va=lambda x: x['mean_va'].astype(float),
                                day_month_mean=lambda x: pd.to_datetime(
                                    x['begin_yr'] + "-" + x['month_nu'].astype(str) + "-" + x['day_nu'].astype(str),
                                    errors="coerce"),
                                day_of_year=lambda x: x['day_month_mean'].dt.dayofyear
                            )
                        elif stat_type == 'max':
                            df = df.assign(
                                month_nu=lambda x: x['month_nu'].astype(int),
                                day_nu=lambda x: x['day_nu'].astype(int),
                                count_nu=lambda x: x['count_nu'].astype(int),
                                max_va=lambda x: x['max_va'].astype(float).astype(int),
                                day_month_max=lambda x: pd.to_datetime(
                                    x['max_va_yr'] + "-" + x['month_nu'].astype(str) + "-" + x['day_nu'].astype(str),
                                    errors="coerce"),
                                day_of_year=lambda x: x['day_month_max'].dt.dayofyear
                            )
                        if df.empty:
                            print(f"No {type} data found for site USGS-{usgs_sensor_id}.")
                            return None

                        return df
                    except pd.errors.EmptyDataError:
                        print(f"Error: No data could be parsed for site USGS-{usgs_sensor_id}.")
                        return None
                else:
                    print(f"No {type} data found for site USGS-{usgs_sensor_id}.")
                    return None
            else:
                print(f"API response returned error: {response.status}")
                return None


async def mean_historical_daily_usgs_sensor_stats_df_async(type, time_period, usgs_sensor_id, start_year, end_year):
    """
    Fetch mean historical daily USGS sensor data asynchronously.
    """
    return await fetch_historical_data(type, time_period, usgs_sensor_id, 'mean', start_year, end_year)


async def max_historical_daily_usgs_sensor_stats_df_async(type, time_period, usgs_sensor_id, start_year, end_year):
    """
    Fetch max historical daily USGS sensor data asynchronously.
    """
    return await fetch_historical_data(type, time_period, usgs_sensor_id, 'max', start_year, end_year)





# ------------------------ MEAN HISTORICAL DATA -----------------

async def optimized_add_mean_historical_stats_to_gdf(type: str, time_period: str, filtered_gdf, start_year: str, end_year: str):
    """
    Add historical mean streamflow data to the filtered GeoDataFrame using batch processing. This function makes fetching
    all results nearly instant. This is the most important part to make async. The previous code could only make 10 requests
    at one time, and would process one after another. This code seeks to speed up the requests by concurrently making
    requests, limiting wait times. This function reduced time to process from ~ 1 min 48 seconds to 10 seconds (-91%)!

    Returns
    --------
    GeoDataframe: 2 added columns: mean historical stats for each sensor and the % difference from the mean each sensor currently is.



     """
    # Get unique site numbers
    unique_sites = filtered_gdf['site_no'].unique()

    # Split the site numbers into batches of 10
    site_batches = np.array_split(unique_sites, np.ceil(len(unique_sites) / 10))



    # Fetch historical data for each batch
    async def fetch_all_mean_batches(type, time_period, site_batches, start_year, end_year):
        tasks = []
        for batch in site_batches:
            batch_sites = ",".join(batch)
            tasks.append(
                mean_historical_daily_usgs_sensor_stats_df_async(type, time_period, batch_sites, start_year, end_year)
            )
        return await asyncio.gather(*tasks)
    historical_data = await fetch_all_mean_batches('streamflow', 'daily', site_batches, '1970', '2024')
    # Combine all historical data into one DataFrame
    if historical_data:
        historical_df = pd.concat(historical_data, ignore_index=True)
    else:
        print("No historical data fetched.")
        return filtered_gdf

    # Merge the historical data with the GeoDataFrame
    # Match on 'site_no' and 'day_of_year'
    merged_gdf = (
        filtered_gdf
        .merge(
            historical_df[['site_no', 'day_of_year', 'mean_va', 'day_month_mean']],
            how='left',
            left_on=['site_no', 'day_of_year'],
            right_on=['site_no', 'day_of_year']
        )
        .rename(columns={'mean_va': 'mean_historical_streamflow_ft3_sec'})
        .drop_duplicates(subset=['site_no', 'day_of_year'])
    )

    # Add the percent difference column
    merged_gdf['percent_difference_streamflow_from_mean'] = merged_gdf.apply(
        lambda row: ((row['discharge_streamflow_ft3_sec'] - row['mean_historical_streamflow_ft3_sec']) / row['mean_historical_streamflow_ft3_sec']) * 100
        if row['mean_historical_streamflow_ft3_sec'] != 0 else None,
        axis=1
    ).round(0)

    return merged_gdf


# ----------------------------- MAX HISTORICAL DATA ----------------------


async def optimized_add_max_historical_stats_to_gdf(type: str, time_period: str, filtered_gdf, start_year: str, end_year: str):
    """
    Add historical max streamflow data to the filtered GeoDataFrame using batch processing. This function makes fetching
    all results nearly instant. This is the most important part to make async. The previous code could only make 10 requests
    at one time, and would process one after another. This code seeks to speed up the requests by concurrently making
    requests, limiting wait times.
    """
    unique_sites = filtered_gdf['site_no'].unique()
    site_batches = np.array_split(unique_sites, np.ceil(len(unique_sites) / 10))


    async def fetch_all_max_batches(type, time_period, site_batches, start_year, end_year):
        '''
        Gather all the historical max api data requests for every USGS site_no in the dataframe.

        '''
        api_requests = []
        for batch in site_batches:
            batch_sites = ",".join(batch)
            api_requests.append(
                max_historical_daily_usgs_sensor_stats_df_async(type, time_period, batch_sites, start_year, end_year)
            )
        return await asyncio.gather(*api_requests) # await pauses the function until ALL api requests are completed
                                                    # once every api call is made, await triggers to perform every api request at once!


     # Here, the await function is waiting for fetch_all_batches to complete
    historical_data = await fetch_all_max_batches('streamflow', 'daily', site_batches, '1970', '2024')

    if historical_data:
        historical_df = pd.concat(historical_data, ignore_index=True)
    else:
        print("No historical data fetched.")
        return filtered_gdf

    merged_gdf = (
        filtered_gdf
        .merge(historical_df[['site_no', 'day_of_year', 'max_va', 'day_month_max']],
               how='left',
               on=['site_no', 'day_of_year'])
        .rename(columns={'max_va': 'max_historical_streamflow_ft3_sec'})
        .drop_duplicates(subset=['site_no', 'day_of_year'])
        .reset_index(drop=True)
    )

    return merged_gdf


# --------------------------------------------- MERGE STREAMFLOW AND GAGE HEIGHT DATA ----------------------

def merge_streamflow_gage_height_gdfs(streamflow_gdf, gage_height_gdf):

    current_streamflow_gage_gdf = (streamflow_gdf
                        .merge(gage_height_gdf[['site_no', 'obs_time', 'obs_result']], on=['site_no', 'obs_time'], how = 'left')
                        .rename(columns = {'obs_result_x' : 'discharge_streamflow_ft3_sec','obs_result_y' : 'gage_height_ft' })
                        .drop(columns = ['loc_name', 'loc_description', 'ds_name', 'ds_description', 'op_name', 'op_description', 'obs_id'])
                        )
    return current_streamflow_gage_gdf






# -------------------------------------- GEOLOCATE SENSOR DATA -----------------------------

def geolocate_usgs_sensors(sensor_gdf):

    '''
    Returns geolocated sensor_gdf. If sjoin intersection returns no result, calculates closest river to sensor.
    Uses river coordinates extracted from a geojson produced from NHD shapefile, downloaded from the USGS portal.

    '''
    # Import GeoJSON containing rivers names and locations


    geojson_path = os.path.join(script_directory, 'NHD_Named_CA_Rivers.geojson')
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
        def get_nearest_river(sensor):
            # Calculate distances from the sensor to all rivers
            distances = NHD_rivers_gpd.geometry.distance(sensor.geometry)
            # Get the name of the nearest river
            nearest_index = distances.idxmin()
            return NHD_rivers_gpd.loc[nearest_index, 'gnis_name']

        # Apply the function to find the nearest river for each sensor
        no_intersection_sensors['gnis_name'] = no_intersection_sensors.apply(get_nearest_river, axis=1)

        # Update the 'gnis_name' column in the main GeoDataFrame with the nearest river names
        geolocated_rivers_gdf.update(no_intersection_sensors[['gnis_name']])

    # Step 4: Rename the 'gnis_name' column to 'nearest_stream_or_river'
    geolocated_rivers_gdf = (geolocated_rivers_gdf
                             .rename(columns={'gnis_name': 'nearest_stream_or_river'})
                             .drop(columns=['index_right', 'thing_id']) # thing_id is a duplicate column of thing_name, index_right is not needed

    )


    return geolocated_rivers_gdf


# ------------------------------------ ADD PERCENTILES --------------------------------------
def add_percentile_streamflows(gdf):
    percentile_df = (pd
                    .read_csv(os.path.join(script_directory, 'streamflow_percentiles.csv'),
                    dtype= {'site_no': 'str'})
    )
    merged_gdf = gdf.merge(percentile_df, on= ['site_no', 'day_of_year'], how = 'left')

    # Add 'percent_difference_from_median' column
    merged_gdf['percent_difference_streamflow_from_median'] = merged_gdf.apply(
        lambda row: ((row['discharge_streamflow_ft3_sec'] - row['50th_percentile']) / row['50th_percentile']) * 100
        if row['50th_percentile'] != 0 else None,
        axis=1
    ).round(0)
    return merged_gdf

# -------------------------------------- APPEND NEW GEOLOCATED SENSOR DATA TO POSTGIS ----------------------
def append_new_sensor_data_to_postgis(gdf):

    """
    Appends new USGS sensor data to the PostGIS-enabled PostgreSQL database on Python-Anywhere.
    Only appends rows where (ds_name, obs_time) combinations do not already exist in the database. Makes connection
    to database, SELECT every ds_name and obs_time values from the database, compares the new geodataframe with existing
    database combinations. Only selects those rows that have unique ds_name/obs_time combinations. Appends unique gdf data
    to database.
    """
    # Server credentials
    db_username = os.getenv('DB_USERNAME')
    db_password =  os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')

    # Create the engine with GeoAlchemy2 support
    engine = create_engine(
        f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    )


    #logging.info("Appending new sensor data to PostGIS database.")
    #print("Appending new sensor data to PostGIS database.")

    try:
        # Query existing data to get `thing_name` and `obs_time` combinations
        with engine.connect() as conn:
            result = conn.execute(
                text('SELECT thing_name, obs_time FROM "usgs_water_sensor_data"')
            )
            # Access tuple elements by index
            existing_data = set((row[0], row[1]) for row in result)

        # Filter out rows from the GeoDataFrame that already exist in the database
        new_data = gdf[~gdf.apply(
            lambda row: (row['thing_name'], row['obs_time']) in existing_data, axis=1
        )]

        # Append the filtered data to the database
        if not new_data.empty:

            # Ensure the geometry column is valid
            new_data['geometry'] = new_data['geometry'].apply(lambda geom: geom if geom is not None else None)
            new_data.to_postgis(
                'usgs_water_sensor_data',  # Table name
                engine,  # SQLAlchemy engine
                if_exists='append',  # Append new data
                index=False,  # Don't write the index
                dtype={'geometry': Geometry('POINT', srid=4326)}  # Define the geometry column type
            )
            #logging.info(f"{len(new_data)} new rows to append to the database!")
            print(f"{len(new_data)} new rows successfully written to the 'usgs_water_sensor_data' table.")
        else:
            print("No new rows to add. All data already exists in the database.")
            #logging.info("No new rows to add. All data already exists in the database.")

    except SQLAlchemyError as e:
        print(f"An error occurred while writing to the database: {e}")
        #logging.error(f"An error occurred while writing to the database: {e}")





def main():
    """
    Main function to fetch, process, and save groundwater data.
    """
    start_time = time.time()
    #logging.info("Main function started.")
    print ("Main function has started")
    try:
        state_name = "California"
        response = get_groundwater_api_json_response(state_name)
        if response:
            #logging.info("Converting JSON response to GeoDataFrame and Geolocating Streams.")
            print("Converting JSON response and filter for Streamflow + Gage Height data to GeoDataFrame ")
            # ---------------- GDF CREATIONS ---------------
            ca_water_gdf = json_to_gdf(response)
            streamflow_filtered_gdf =  ca_water_gdf[(ca_water_gdf['op_name'] == 'Discharge, cubic feet per second')]
            gage_height_filtered_gdf = ca_water_gdf[(ca_water_gdf['op_name'] == 'Gage height, feet')]
            current_streamflow_gage_gdf = merge_streamflow_gage_height_gdfs(streamflow_filtered_gdf, gage_height_filtered_gdf)

            print("Retrieving historical mean and max streamflow data from 1970 to 2024")
            mean_gdf = asyncio.run(optimized_add_mean_historical_stats_to_gdf(type = 'streamflow',
                                           time_period= 'daily',
                                           filtered_gdf = current_streamflow_gage_gdf,
                                           start_year = '1970', end_year = '2024')
            )
            mean_max_gdf = asyncio.run(optimized_add_max_historical_stats_to_gdf(type = 'streamflow',
                                                                     time_period = 'daily',
                                                                     filtered_gdf = mean_gdf ,
                                                                     start_year = '1970',
                                                                       end_year = '2023'))

            print("Geolocating USGS sensors.")
            geolocated_mean_max_gdf = geolocate_usgs_sensors(mean_max_gdf)

            print("Adding Percentiles to GDF")
            percentiles_added_gdf = add_percentile_streamflows(geolocated_mean_max_gdf)



            print("Appending new sensor data to PostGIS database.")
            append_new_sensor_data_to_postgis(percentiles_added_gdf)



            # Stop the timer
            end_time = time.time()

            # Calculate elapsed time
            elapsed_time = end_time - start_time


            #print(county_merged_sensor_gdf)
            print(f"This script was successful and took : {elapsed_time:.2f} seconds.")



    except Exception as e:
        #logging.error(f"An error occurred in main(): {e}")
        print(f"An error occurred:{e} ")




if __name__ == "__main__":
    main()


# COLUMNS:
# 'thing_id', 'thing_name', 'thing_description',
 #      'discharge_streamflow_ft3/sec', 'obs_time', 'geometry', 'site_no',
  #     'day_of_year', 'gage_height_ft', 'mean_historical_streamflow_ft3/sec',
   #    'day_month_mean', 'percent_difference_streamflow_from_mean',
    #   'max_historical_streamflow_ft3/sec', 'day_month_max',
     #  'nearest_stream_or_river']

# DTYPES
#thing_name                                                     object
#thing_description                                              object
#discharge_streamflow_ft3/sec                                    int64
#obs_time                                   datetime64[ns, US/Pacific]
#geometry                                                     geometry
#site_no                                                        object
#day_of_year                                                     int32
#gage_height_ft                                                float64
#mean_historical_streamflow_ft3/sec                            float64
#day_month_mean                                         datetime64[ns]
#percent_difference_streamflow_from_mean                       float64
#max_historical_streamflow_ft3/sec                             float64
#day_month_max                                          datetime64[ns]
#nearest_stream_or_river                                        object
