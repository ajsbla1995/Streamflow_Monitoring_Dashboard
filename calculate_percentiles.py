import requests
import pandas as pd
import numpy as np
from io import StringIO


def fetch_historical_data(type, time_period, usgs_sensor_id, stat_type, start_year, end_year):
    """
    Fetch historical USGS sensor data using requests.

    Args:
        type (str): Type of data ('streamflow' or 'gage_height').
        time_period (str): Time period ('daily', 'monthly', etc.).
        usgs_sensor_id (str): USGS site ID.
        stat_type (str): Statistical type ('mean' or 'max').
        start_year (str): Start year.
        end_year (str): End year.

    Returns:
        pd.DataFrame: Processed dataframe or None if no data available.
    """
    usgs_base_url = 'https://waterservices.usgs.gov/nwis/stat/'

    # Determine parameter code
    parameter = {
        'streamflow': "00060",
        'gage_height': "00065"
    }.get(type, None)

    if parameter is None:
        print(f"Invalid type: {type}. Must be 'streamflow' or 'gage_height'.")
        return None

    params = {
        "sites": usgs_sensor_id,
        "statReportType": time_period,
        "statType": stat_type,
        "parameterCd": parameter,
        "startDt": start_year,
        "endDt": end_year,
        "format": "rdb"
    }

    try:
        response = requests.get(usgs_base_url, params=params, timeout=30)
        if response.status_code == 200 and response.text.strip():
            try:
                # Load the data into a pandas DataFrame
                data = StringIO(response.text)
                df = pd.read_csv(data, sep="\t", comment="#", engine="python").drop(index=0)

                # Process and clean the data
                df = df.rename(columns=str.strip)
                if stat_type == "mean":
                    df = df.assign(
                        month_nu=lambda x: x['month_nu'].astype(int),
                        day_nu=lambda x: x['day_nu'].astype(int),
                        mean_va=lambda x: pd.to_numeric(x['mean_va'], errors='coerce'),
                        day_of_year=lambda x: pd.to_datetime(
                            x['begin_yr'] + "-" + x['month_nu'].astype(str) + "-" + x['day_nu'].astype(str),
                            errors="coerce"
                        ).dt.dayofyear
                    )
                elif stat_type == "max":
                    df = df.assign(
                        month_nu=lambda x: x['month_nu'].astype(int),
                        day_nu=lambda x: x['day_nu'].astype(int),
                        max_va=lambda x: pd.to_numeric(x['max_va'], errors='coerce'),
                        day_of_year=lambda x: pd.to_datetime(
                            x['max_va_yr'] + "-" + x['month_nu'].astype(str) + "-" + x['day_nu'].astype(str),
                            errors="coerce"
                        ).dt.dayofyear
                    )

                # Drop rows without valid `day_of_year`
                df = df.dropna(subset=["day_of_year"]).reset_index(drop=True)

                return df
            except pd.errors.EmptyDataError:
                print(f"No data could be parsed for site {usgs_sensor_id}.")
                return None
        else:
            print(f"API response error {response.status_code}: {response.reason}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def get_10yr_mean_streamflow(type, usgs_sensor_id, time_period, stat_type, start_year, end_year):
   
    '''
    Returns a range of years in which we find the mean for each day of the year, if available. For instance, if we say the
    start_year is = 2015 and the end_year = 2024, for every year between these dates, we find all the mean streamflows. We then
    merge the daily data for all these years into a single dataframe. Then we add a column of the sensor id so that we can use the 
    df later on to find percentiles.
    
    '''
    all_data = []

    # Convert start_year and end_year to integers for range
    start_year = int(start_year)
    end_year = int(end_year)

    for year in range(start_year, end_year + 1):  # Loop through years
        print(f"Fetching data for year {year}...")
        yearly_data = fetch_historical_data(
            type=type,
            time_period=time_period,
            stat_type=stat_type,
            usgs_sensor_id=usgs_sensor_id,
            start_year=str(year),  # Convert year back to string for the API
            end_year=str(year)
        )
        if yearly_data is not None:
            yearly_data = yearly_data[['day_of_year','mean_va']].rename(columns={"mean_va": f"mean_va_{year}"})
            all_data.append(yearly_data)

    if not all_data:
        print("No data retrieved for the given years.")
        return None

    # Merge all yearly data on day_of_year
    merged_df = pd.DataFrame({"day_of_year": range(1, 367)})  # Handles leap years
    for yearly_data in all_data:
        merged_df = pd.merge(merged_df, yearly_data, on=["day_of_year"], how="left")

    merged_df['site_no']= usgs_sensor_id # Add back the site_no as a column
    return merged_df





def calculate_percentiles(type, time_period, usgs_sensor_id, stat_type, start_year, end_year):
    # Retrieve historical data
    historical_df = get_10yr_mean_streamflow(type, 
                                             usgs_sensor_id, 
                                             time_period, 
                                             stat_type, 
                                             start_year, 
                                             end_year)

    if historical_df is None or historical_df.empty:
        print(f"No historical data available for sensor {usgs_sensor_id}.")
        return historical_df

    # Select the columns that represent the streamflow for each year
    year_columns = [col for col in historical_df.columns if col.startswith('mean_va_')]

    # Calculate percentiles across the year columns (ignoring NaN values)
    def calculate_row_percentiles(row):
        valid_values = row[1:].dropna()  # Exclude NaN values for percentile calculation
        if valid_values.empty:
            return [np.nan] * 5  # Return NaN if no valid data exists
        return [
            np.percentile(valid_values, 10),
            np.percentile(valid_values, 25),
            np.percentile(valid_values, 50),
            np.percentile(valid_values, 75),
            np.percentile(valid_values, 90),
        ]

    percentiles_df = historical_df[['day_of_year'] + year_columns].apply(calculate_row_percentiles, axis=1)

    # Create a new DataFrame with the calculated percentiles
    percentiles_df = pd.DataFrame(percentiles_df.tolist(), columns=[
        '10th_percentile', 
        '25th_percentile', 
        '50th_percentile', 
        '75th_percentile', 
        '90th_percentile'
    ])

    # Add the 'day_of_year' column back to the percentiles_df
    percentiles_df['day_of_year'] = historical_df['day_of_year'].values
    percentiles_df['site_no'] = usgs_sensor_id

    return percentiles_df.drop_duplicates()




def percentiles_for_all_unique_sensors(gdf, start_year, end_year):
    unique_sensors = gdf['site_no'].unique()

    all_percentiles_list = []
    for site_no in unique_sensors:
        all_percentiles_gdf = calculate_percentiles(type = 'streamflow', 
                                  usgs_sensor_id = site_no, 
                                  time_period = 'daily',
                                  stat_type = 'mean',
                                  #original_df = gdf, 
                                  start_year= start_year, 
                                  end_year= end_year)
        # Add the site_no as a column
          # Check if the result is not None and process further
        if all_percentiles_gdf is not None:
            # Add the site_no as a column
            all_percentiles_gdf['site_no'] = site_no
            # Append the result to the list
            all_percentiles_list.append(all_percentiles_gdf)
        else:
            print(f"No data available for site_no: {site_no}")
        #all_percentiles_gdf['site_no'] = site_no
        #all_percentiles_list.append(all_percentiles_gdf)
    
        # Optionally: Combine all the results into a single DataFrame
    combined_percentiles_df = pd.concat(all_percentiles_list)
    return combined_percentiles_df
        
