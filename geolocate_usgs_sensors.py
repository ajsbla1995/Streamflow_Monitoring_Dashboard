import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from IPython.display import Image
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
from io import StringIO
import matplotlib.pyplot as plt
import pull_data_from_local_postgisDB
import get_historical_usgs_sensor_data 
import make_pydeck 
import plotly.express as px
import plotly.graph_objects as go


flowline0_shape_path =  "/Users/andresabala/Downloads/Data Analysis Projects/Ground_Water_Dashboard/Scripts/Data/NHD_H_California_State_Shape/Shape/NHDFlowline_0.shp"

flowline0_gdf = (gpd
                 .read_file(flowline0_shape_path)
                 .to_crs("EPSG:4236")
)
flowline0_gdf

nhd_named_rivers= flowline0_gdf[flowline0_gdf['gnis_name'].notnull()]
nhd_named_rivers

# Save as GeoJSON
output_geojson_path = "/Users/andresabala/Downloads/Data Analysis Projects/Ground_Water_Dashboard/Scripts/Data/NHD_Named_CA_Rivers.geojson"
nhd_named_rivers.to_file(output_geojson_path, driver="GeoJSON")


