# Databricks notebook source

API_TOKEN = dbutils.secrets.get(scope="noaa_api_scope", key="noaa_api_key")  


print("API token retrieved successfully!")

# COMMAND ----------

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# NOAA API Endpoint and API Token
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
HEADERS = {"token": API_TOKEN} 

# Parameters
DATASET_ID = "GHCND"  
START_YEAR = 1950      # Start year
END_YEAR = 2024       # End year
LIMIT = 1000           # Max rows per request
EXPECTED_ROWS = 100000  
STATIONS = [
    "GHCND:USW00094728",  
    "GHCND:USW00023234", 
    "GHCND:USW00023174", 
    "GHCND:USW00012960",  
    "GHCND:USW00013874"  
]


COLUMNS_MAPPING = {
    "TMAX": "max_temperature",
    "TMIN": "min_temperature",
    "PRCP": "precipitation",
    "SNOW": "snowfall",
    "SNWD": "snow_depth",
    "AWND": "avg_wind_speed",
    "WSF2": "fastest_2min_wind",
    "WDF2": "wind_direction_2min",
    "TAVG": "avg_temperature",
    "WT01": "weather_type_1"
}


def fetch_data_for_station_year(station_id, year):
    observations_dict = {}
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    offset = 1

    while True:
        params = {
            "datasetid": DATASET_ID,
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "limit": LIMIT,
            "offset": offset,
            "units": "metric"
        }
        response = requests.get(BASE_URL, headers=HEADERS, params=params)

        # Error handling
        if response.status_code != 200:
            print(f"Error: {response.status_code}. {response.text}")
            break

        results = response.json().get("results", [])
        if not results:
            break  

        # Process raw records
        for record in results:
            datatype = record.get("datatype")
            if datatype in COLUMNS_MAPPING:  
                date = record.get("date")
                key = (date, station_id)  # Grouping key

                if key not in observations_dict:
                    observations_dict[key] = {
                        "date": date,
                        "station": station_id,
                        "latitude": record.get("latitude"),
                        "longitude": record.get("longitude")
                    }

                # Add the value for the specific datatype
                observations_dict[key][COLUMNS_MAPPING[datatype]] = record.get("value")


        offset += LIMIT
        time.sleep(1)  # Avoid hitting rate limits

    return list(observations_dict.values())

# Main script to fetch and process data
all_data = []
current_percent = 0  # Track progress percentage

print("Fetching NOAA data...\n")
for station in STATIONS:
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching data for station: {station}, year: {year}")
        data = fetch_data_for_station_year(station, year)

        # Add non-empty results directly
        if data:
            print(f"Data successfully retrieved for station: {station}, year: {year}")
            all_data.extend(data)

if all_data:
    df = pd.DataFrame(all_data)

    # Drop duplicate rows
    total_rows_before = len(df)
    df.drop_duplicates(inplace=True)
    total_rows_after = len(df)


    for i in range(1, 101):
        threshold = (EXPECTED_ROWS * i) // 100
        if total_rows_after >= threshold and current_percent < i:
            print(f"Progress: {i}% complete - {total_rows_after} rows added.")
            current_percent = i

    # Save to CSV
    df.to_csv("NOAA_weather_data_1950_2024.csv", index=False)
    print(f"\nData fetching complete!")
    print(f"Total rows before removing duplicates: {total_rows_before}")
    print(f"Total rows after removing duplicates: {total_rows_after}")
    print("Data saved to 'noaa_cleaned_weather_data.csv'")

    
    print("\nCleaned data preview:")
    print(df.head())
else:
    print("No data fetched.")


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze Layer

# COMMAND ----------

data=pd.read_csv("NOAA_weather_data_1950_2024.csv")

# COMMAND ----------

data.shape

# COMMAND ----------

data.columns

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


spark = SparkSession.builder \
    .appName("Load Existing DataFrame into Spark") \
    .getOrCreate()


schema = StructType([
    StructField("date", StringType(), True),
    StructField("station", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("precipitation", DoubleType(), True),
    StructField("snowfall", DoubleType(), True),
    StructField("snow_depth", DoubleType(), True),
    StructField("max_temperature", DoubleType(), True),
    StructField("min_temperature", DoubleType(), True),
    StructField("avg_wind_speed", DoubleType(), True),
    StructField("wind_direction_2min", IntegerType(), True),
    StructField("fastest_2min_wind", DoubleType(), True),
    StructField("weather_type_1", StringType(), True),
    StructField("avg_temperature", DoubleType(), True)
])



spark_data = spark.createDataFrame(data)



spark_data.printSchema()


# COMMAND ----------


num_rows = spark_data.count()


num_columns = len(spark_data.columns)

print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_columns}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Silver Layer

# COMMAND ----------

from pyspark.sql.functions import col, count, when

display(spark_data.limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC # The longitude and Latitude can be reterive  from the API as (We can't fill with any way)

# COMMAND ----------

import requests
import pandas as pd


# List of station IDs
STATIONS = [
    "GHCND:USW00094728",  # New York Central Park
    "GHCND:USW00023234",  # Los Angeles International Airport
    "GHCND:USW00023174",  # San Francisco International Airport
    "GHCND:USW00012960",  # Chicago O'Hare International
    "GHCND:USW00013874"   # Seattle-Tacoma International Airport
]


api_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"


headers = {
    "token":  API_TOKEN
}

# Function to get station metadata
def get_station_metadata(stations):
    data = []
    for station_id in stations:
        response = requests.get(f"{api_url}/{station_id}", headers=headers)
        if response.status_code == 200:
            station_data = response.json()
            data.append({
                "station_id": station_id,
                "name": station_data.get("name"),
                "latitude": station_data.get("latitude"),
                "longitude": station_data.get("longitude")
            })
        else:
            print(f"Failed to fetch data for {station_id}: {response.status_code}")
    return pd.DataFrame(data)


station_df = get_station_metadata(STATIONS)


display(station_df)


# COMMAND ----------

# Save station_df as a CSV file
station_df.to_csv("station_data.csv", index=False)


# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType


schema = StructType([
    StructField("station_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True)
])


spark_location = spark.createDataFrame(station_df, schema=schema)



spark_location .printSchema()


# COMMAND ----------


spark_data = spark_data.drop("latitude", "longitude")

display(spark_data.limit(10))

# COMMAND ----------


spark_data_alias = spark_data.alias("data")
spark_location_alias = spark_location.alias("location")


joined_df = spark_data_alias.join(
    spark_location_alias,
    spark_data_alias.station == spark_location_alias.station_id,
    "left"
)


spark_data = joined_df.select(
    "data.*",
    "location.latitude",
    "location.longitude"
)

display(spark_data.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### fill null values in the avg_wind_speedn & wind_direction_2min c column based on the yearly average for each location (latitude, longitude), or fallback to 0 when no average is found

# COMMAND ----------

from pyspark.sql import functions as F

# Step 1: Extract year from the date column
spark_data = spark_data.withColumn("year", F.year("date"))


averages_by_year_location = spark_data.groupBy(
    "year", "latitude", "longitude"
).agg(
    F.avg("avg_wind_speed").alias("avg_wind_speed_avg"),
    F.avg("wind_direction_2min").alias("wind_direction_avg")
)


spark_data = spark_data.alias("data").join(
    averages_by_year_location.alias("avg"),
    (F.col("data.year") == F.col("avg.year")) &
    (F.col("data.latitude") == F.col("avg.latitude")) &
    (F.col("data.longitude") == F.col("avg.longitude")),
    "left"
)


spark_data = spark_data.withColumn(
    "avg_wind_speed",
    F.when(F.col("data.avg_wind_speed").isNotNull(), F.col("data.avg_wind_speed"))
     .when(F.col("avg.avg_wind_speed_avg").isNotNull(), F.col("avg.avg_wind_speed_avg"))
     .otherwise(0)
).withColumn(
    "wind_direction_2min",
    F.when(F.col("data.wind_direction_2min").isNotNull(), F.col("data.wind_direction_2min"))
     .when(F.col("avg.wind_direction_avg").isNotNull(), F.col("avg.wind_direction_avg"))
     .otherwise(0)
)


columns_to_keep = [
    "data.date",
    "data.station",
    "data.latitude",
    "data.longitude",
    "data.precipitation",
    "data.snowfall",
    "data.snow_depth",
    "data.max_temperature",
    "data.min_temperature",
    "avg_wind_speed",  # Updated column
    "wind_direction_2min",  # Updated column
    "data.fastest_2min_wind",
    "data.weather_type_1",
    "data.avg_temperature"
]

spark_data = spark_data.selectExpr(*columns_to_keep)


spark_data.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ## avg_temperature column with the average of min_temperature and max_temperature if avg_temperature is null

# COMMAND ----------

from pyspark.sql import functions as F


spark_data = spark_data.withColumn(
    "avg_temperature",
    F.when(F.col("avg_temperature").isNotNull(), F.col("avg_temperature"))
     .when((F.col("min_temperature").isNotNull()) & (F.col("max_temperature").isNotNull()),
           (F.col("min_temperature") + F.col("max_temperature")) / 2)
     .otherwise(0)
)



# COMMAND ----------

# MAGIC %md
# MAGIC > ## #  fastest_2min_wind 

# COMMAND ----------

from pyspark.sql import functions as F

spark_data = spark_data.fillna({"fastest_2min_wind": 0})



# COMMAND ----------

# MAGIC %md
# MAGIC # weather_type_1

# COMMAND ----------


unique_weather_types = spark_data.select("weather_type_1").distinct()

unique_weather_types.show(truncate=False)


# COMMAND ----------

from pyspark.sql import functions as F


spark_data = spark_data.fillna({"weather_type_1": 0})


# COMMAND ----------



# COMMAND ----------

display(spark_data.limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Date Column 
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

spark_data = spark_data.withColumn("Date_1", F.to_date("date", "yyyy-MM-dd'T'HH:mm:ss"))

spark_data.select("date", "Date_1").show(10)


# COMMAND ----------

display(spark_data.limit(10))

# COMMAND ----------

from pyspark.sql.functions import round

s
spark_data = spark_data.withColumn("avg_temperature_rounded", round("avg_temperature", 2))


display(spark_data.select("avg_temperature", "avg_temperature_rounded").limit(10))

# COMMAND ----------

spark_data = spark_data.drop("avg_temperature")


display(spark_data.limit(10))

# COMMAND ----------

display(spark_data.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Gold Layer
# MAGIC

# COMMAND ----------

station_mapping = {
    "GHCND:USW00094728": "New York City Central Park, NY",
    "GHCND:USW00023234": "Los Angeles International Airport, CA",
    "GHCND:USW00023174": "San Francisco International Airport, CA",
    "GHCND:USW00012960": "Houston Intercontinental Airport, TX",
    "GHCND:USW00013874": "Atlanta Hartsfield Airport, GA"
}

# COMMAND ----------

import pandas as pd
import plotly.graph_objects as go


first_station = STATIONS[0] 
filtered_data = spark_data.filter(spark_data.station == first_station)


temperature_data = filtered_data.select(
    "Date_1", "avg_temperature_rounded", "max_temperature", "min_temperature"
).orderBy("Date_1")

pandas_data = temperature_data.toPandas()

pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['avg_temperature_rounded'], 
    mode='lines',
    name='Average Temperature',
    line=dict(width=2)
))


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['max_temperature'], 
    mode='lines',
    name='Max Temperature',
    line=dict(dash='dash', width=2)
))


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['min_temperature'], 
    mode='lines',
    name='Min Temperature',
    line=dict(dash='dot', width=2)
))


fig.update_layout(
    title=f"Temperature Trends for Station: {first_key[0]}",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    xaxis=dict(
        rangeslider=dict(visible=True), 
        type="date"
    ),
    legend=dict(x=0, y=1, traceorder="normal"),
    template="plotly_white"
)


fig.show()


# COMMAND ----------

import pandas as pd
import plotly.graph_objects as go

first_station = STATIONS[1]  
filtered_data = spark_data.filter(spark_data.station == first_station)


temperature_data = filtered_data.select(
    "Date_1", "avg_temperature_rounded", "max_temperature", "min_temperature"
).orderBy("Date_1")

pandas_data = temperature_data.toPandas()


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['avg_temperature_rounded'], 
    mode='lines',
    name='Average Temperature',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['max_temperature'], 
    mode='lines',
    name='Max Temperature',
    line=dict(dash='dash', width=2)
))


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['min_temperature'], 
    mode='lines',
    name='Min Temperature',
    line=dict(dash='dot', width=2)
))


fig.update_layout(
    title=f"Temperature Trends for Station: {first_key[1]}",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    xaxis=dict(
        rangeslider=dict(visible=True), 
        type="date"
    ),
    legend=dict(x=0, y=1, traceorder="normal"),
    template="plotly_white"
)

fig.show()


# COMMAND ----------

import pandas as pd
import plotly.graph_objects as go


first_station = STATIONS[2] 
filtered_data = spark_data.filter(spark_data.station == first_station)


temperature_data = filtered_data.select(
    "Date_1", "avg_temperature_rounded", "max_temperature", "min_temperature"
).orderBy("Date_1")


pandas_data = temperature_data.toPandas()


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])

fig = go.Figure()


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['avg_temperature_rounded'], 
    mode='lines',
    name='Average Temperature',
    line=dict(width=2)
))


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['max_temperature'], 
    mode='lines',
    name='Max Temperature',
    line=dict(dash='dash', width=2)
))

fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['min_temperature'], 
    mode='lines',
    name='Min Temperature',
    line=dict(dash='dot', width=2)
))


fig.update_layout(
    title=f"Temperature Trends for Station: {first_key[2]}",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    xaxis=dict(
        rangeslider=dict(visible=True),  
        type="date"
    ),
    legend=dict(x=0, y=1, traceorder="normal"),
    template="plotly_white"
)

fig.show()


# COMMAND ----------

import pandas as pd
import plotly.graph_objects as go


first_station = STATIONS[3]  
filtered_data = spark_data.filter(spark_data.station == first_station)

temperature_data = filtered_data.select(
    "Date_1", "avg_temperature_rounded", "max_temperature", "min_temperature"
).orderBy("Date_1")


pandas_data = temperature_data.toPandas()


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])


fig = go.Figure()


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['avg_temperature_rounded'], 
    mode='lines',
    name='Average Temperature',
    line=dict(width=2)
))


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['max_temperature'], 
    mode='lines',
    name='Max Temperature',
    line=dict(dash='dash', width=2)
))


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['min_temperature'], 
    mode='lines',
    name='Min Temperature',
    line=dict(dash='dot', width=2)
))


fig.update_layout(
    title=f"Temperature Trends for Station: {first_key[3]}",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    xaxis=dict(
        rangeslider=dict(visible=True),  
        type="date"
    ),
    legend=dict(x=0, y=1, traceorder="normal"),
    template="plotly_white"
)


fig.show()


# COMMAND ----------

import pandas as pd
import plotly.graph_objects as go


first_station = STATIONS[4] 
filtered_data = spark_data.filter(spark_data.station == first_station)


temperature_data = filtered_data.select(
    "Date_1", "avg_temperature_rounded", "max_temperature", "min_temperature"
).orderBy("Date_1")

pandas_data = temperature_data.toPandas()


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])

fig = go.Figure()


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['avg_temperature_rounded'], 
    mode='lines',
    name='Average Temperature',
    line=dict(width=2)
))


fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['max_temperature'], 
    mode='lines',
    name='Max Temperature',
    line=dict(dash='dash', width=2)
))

# Add Min Temperature
fig.add_trace(go.Scatter(
    x=pandas_data['Date_1'], 
    y=pandas_data['min_temperature'], 
    mode='lines',
    name='Min Temperature',
    line=dict(dash='dot', width=2)
))


fig.update_layout(
    title=f"Temperature Trends for Station: {first_key[4]}",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    xaxis=dict(
        rangeslider=dict(visible=True),  
        type="date"
    ),
    legend=dict(x=0, y=1, traceorder="normal"),
    template="plotly_white"
)


fig.show()


# COMMAND ----------

import plotly.express as px
import pandas as pd


pandas_data = spark_data.select(
    "Date_1", "latitude", "longitude", "avg_wind_speed", "avg_temperature_rounded", "station"
).dropna().toPandas()

pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])
pandas_data['month_year'] = pandas_data['Date_1'].dt.to_period("M").astype(str)


pandas_data = pandas_data.rename(columns={
    "avg_wind_speed": "Average Wind Speed",
    "avg_temperature_rounded": "Average Temperature (°C)",
    "station": "Station Name"
})


fig = px.scatter_geo(
    pandas_data,
    lon="longitude",
    lat="latitude",
    text="Station Name",
    animation_frame="month_year", 
    size="Average Wind Speed",  
    size_max=50, 
    color="Average Temperature (°C)",  
    hover_name="Station Name",
    hover_data={
        "Average Temperature (°C)": True,
        "Average Wind Speed": True,
        "longitude": False,
        "latitude": False
    },
    title="Interactive Map of Average Temperature and Wind Speed (USA by Month-Year)",
    color_continuous_scale="Viridis",
    projection="albers usa"  
)


fig.update_geos(
    scope="usa",
    showland=True,
    landcolor="rgb(243, 243, 243)",
    subunitcolor="rgb(150, 150, 150)",  
    showsubunits=True,  
    showlakes=True,
    lakecolor="rgb(173, 216, 230)",  
    showrivers=True,
    rivercolor="rgb(204, 204, 204)"
)

fig.update_layout(
    coloraxis_colorbar=dict(title="Average Temperature (°C)"),
    margin={"r":0, "t":50, "l":0, "b":0}
)

fig.show()


# COMMAND ----------

import plotly.express as px
import pandas as pd


pandas_data = spark_data.select(
    "Date_1", "latitude", "longitude", "avg_wind_speed", "avg_temperature_rounded", "station"
).dropna().toPandas()


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])
pandas_data['month_year'] = pandas_data['Date_1'].dt.to_period("M").astype(str)


pandas_data_2024 = pandas_data[pandas_data['Date_1'].dt.year == 2024]


pandas_data_2024 = pandas_data_2024.rename(columns={
    "avg_wind_speed": "Average Wind Speed",
    "avg_temperature_rounded": "Average Temperature (°C)",
    "station": "Station Name"
})


fig = px.scatter_geo(
    pandas_data_2024,
    lon="longitude",
    lat="latitude",
    text="Station Name",
    animation_frame="month_year",  
    size="Average Wind Speed",  
    size_max=50,  
    color="Average Temperature (°C)",  
    hover_name="Station Name",
    hover_data={
        "Average Temperature (°C)": True,
        "Average Wind Speed": True,
        "longitude": False,
        "latitude": False
    },
    title="Interactive Map of Average Temperature and Wind Speed (USA for 2024)",
    color_continuous_scale="Viridis",
    projection="albers usa"
)


fig.update_geos(
    scope="usa",
    showland=True,
    landcolor="rgb(243, 243, 243)",
    subunitcolor="rgb(150, 150, 150)",  
    showsubunits=True, 
    showlakes=True,
    lakecolor="rgb(173, 216, 230)", 
    showrivers=True,
    rivercolor="rgb(204, 204, 204)"
)

fig.update_layout(
    coloraxis_colorbar=dict(title="Average Temperature (°C)"),
    margin={"r":0, "t":50, "l":0, "b":0}
)


fig.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


pandas_data = spark_data.select(
    "Date_1", "avg_temperature_rounded"
).dropna().toPandas()


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])


pandas_data['year'] = pandas_data['Date_1'].dt.year  
yearly_avg_temp = pandas_data.groupby('year', as_index=False).agg({
    'avg_temperature_rounded': 'mean'  
})


x = yearly_avg_temp['year']
y = yearly_avg_temp['avg_temperature_rounded']


coefficients = np.polyfit(x, y, 1) 
linear_trend = np.poly1d(coefficients)
y_trend = linear_trend(x)  

plt.figure(figsize=(12, 6))


plt.plot(x, y, color='tab:blue', linestyle='-', linewidth=2, marker='o', label="Average Temperature")


plt.plot(x, y_trend, color='red', linestyle='--', linewidth=2, label="Linear Trend Line")


plt.title("Average Temperature from 1950 to 2024 with Linear Trend", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Temperature (°C)", fontsize=12)
plt.xticks(range(1950, 2025, 5))  
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()  
plt.tight_layout()

plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt


pandas_data = spark_data.select(
    "Date_1", "avg_temperature_rounded", "station"
).dropna().toPandas()


station_mapping = {
    "GHCND:USW00094728": "Central Park, NY",
    "GHCND:USW00023234": "Los Angeles, CA",
    "GHCND:USW00023174": "San Francisco, CA",
    "GHCND:USW00012960": "Houston, TX",
    "GHCND:USW00013874": "Atlanta, GA"
}
pandas_data['station'] = pandas_data['station'].replace(station_mapping)


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])
pandas_data['month'] = pandas_data['Date_1'].dt.month 


monthly_avg_temp = pandas_data.groupby(['station', 'month'], as_index=False).agg({
    'avg_temperature_rounded': 'mean'
})


plt.figure(figsize=(12, 6))

for station in station_mapping.values():  
    station_data = monthly_avg_temp[monthly_avg_temp['station'] == station]
    plt.plot(
        station_data['month'], 
        station_data['avg_temperature_rounded'], 
        marker='o',
        linestyle='-',
        linewidth=2,
        label=station  
    )


plt.title("Average Monthly Temperature by Station", fontsize=16)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Average Temperature (°C)", fontsize=12)
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.legend(title="Station", loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()


plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt


pandas_data = spark_data.select(
    "Date_1", "precipitation", "station"
).dropna().toPandas()


station_mapping = {
    "GHCND:USW00094728": "Central Park, NY",
    "GHCND:USW00023234": "Los Angeles, CA",
    "GHCND:USW00023174": "San Francisco, CA",
    "GHCND:USW00012960": "Houston, TX",
    "GHCND:USW00013874": "Atlanta, GA"
}
pandas_data['station'] = pandas_data['station'].replace(station_mapping)


pandas_data['Date_1'] = pd.to_datetime(pandas_data['Date_1'])
pandas_data['month'] = pandas_data['Date_1'].dt.month  


monthly_avg_precip = pandas_data.groupby(['station', 'month'], as_index=False).agg({
    'precipitation': 'mean'
})


plt.figure(figsize=(12, 6))


for station in station_mapping.values(): 
    station_data = monthly_avg_precip[monthly_avg_precip['station'] == station]
    plt.plot(
        station_data['month'], 
        station_data['precipitation'], 
        marker='o',
        linestyle='-',
        linewidth=2,
        label=station  
    )


plt.title("Average Monthly Precipitation by Station", fontsize=16)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Average Precipitation (mm)", fontsize=12)
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.legend(title="Station", loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()


plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pandas_data = spark_data.select(
    "avg_temperature_rounded", "precipitation", "station"
).dropna().toPandas()


station_mapping = {
    "GHCND:USW00094728": "Central Park, NY",
    "GHCND:USW00023234": "Los Angeles, CA",
    "GHCND:USW00023174": "San Francisco, CA",
    "GHCND:USW00012960": "Houston, TX",
    "GHCND:USW00013874": "Atlanta, GA"
}
pandas_data['station'] = pandas_data['station'].replace(station_mapping)


pandas_data['precipitation'] = pd.to_numeric(pandas_data['precipitation'], errors='coerce')


plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='precipitation', 
    y='avg_temperature_rounded',
    hue='station', 
    palette='viridis',
    s=100,  
    alpha=0.7,
    data=pandas_data
)
plt.title("Scatter Plot of Precipitation vs Temperature")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Average Temperature (°C)")
plt.legend(title="Station")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()


correlation = pandas_data['precipitation'].corr(pandas_data['avg_temperature_rounded'])
print(f"Correlation between Precipitation and Temperature: {correlation:.2f}")

plt.show()


# COMMAND ----------

# Save the cleaned Spark DataFrame as a CSV file
spark_data.write.csv(
    "/Workspace/Users/thotas95@students.rowan.edu/cleaned_weather_data_1950-2024.csv",
    header=True,  # Include column headers
    mode="overwrite"  # Overwrite the file if it already exists
)

# COMMAND ----------

# Step 1: Convert Spark DataFrame to Pandas DataFrame
pandas_data = spark_data.toPandas()

# Step 2: Save Pandas DataFrame as a CSV File
pandas_data.to_csv("cleaned_weather_data_1950-2024.csv", index=False)

# Confirmation
print("File saved as 'cleaned_weather_data_1950-2024.csv'")

