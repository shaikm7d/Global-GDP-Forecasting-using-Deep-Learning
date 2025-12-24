# Import necessary libraires
import wbdata
import pandas as pd
import numpy as np
import datetime

def download_world_bank_data(indicators, country_codes, start_year, end_year):
    """
    Downloads tabular and time-series data from the World Bank API and returns it in-memory.

    Parameters:
        indicators (dict): A dictionary mapping indicator codes to column names.
        country_codes (list): List of country codes to include.
        start_year (int): Start year for the data.
        end_year (int): End year for the data.

    Returns:
        pandas.DataFrame: The requested World Bank data in a DataFrame.
    """
    # Define the date range for the API call
    data_date_range = (datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 12, 31))
    
    print("Fetching data from the World Bank API")
    # Fetch data using wbdata; this returns a pandas DataFrame
    data = wbdata.get_dataframe(indicators, country=country_codes, date=data_date_range, parse_dates=True, skip_cache=True)
    
    # Reset the index so that 'country' and 'date' become columns in the DataFrame
    data.reset_index(inplace=True)
    
    print("Data loaded into memory")
    return data


# Define the indicators of interest. Note that each indicator code is unique.
indicators = {
    "NY.GDP.PCAP.PP.KD": "GDPpc_2017$",         # GDP per capita, PPP (constant 2017 US$)
    "SP.POP.TOTL": "Population_total",
    "SP.DYN.LE00.IN": "Life_expectancy",
    "SE.ADT.LITR.ZS": "Literacy_rate",
    "SL.UEM.TOTL.ZS": "Unemploymlent_rate",
    "SP.DYN.TFRT.IN": "Fertility_rate",
    "SI.POV.NAHC": "Poverty_ratio",
    "SE.PRM.ENRR": "Primary_school_enrolmet_rate",
    "EG.USE.PCAP.KG.OE": "Energy_use",
    "NE.EXP.GNFS.KD": "Exports_2017$"
}

# Define a regular expression to filter out aggregate regions and non-country entries.
expression = (
    '^(?!(?:World|Upper middle income|Sub\\-Saharan Africa excluding South Africa|Sub\\-Saharan Africa|'
    'Latin America \\& Caribbean|IDA countries in Sub\\-Saharan Africa classified as fragile situations|'
    'IDA\\scountries\\sin\\sSub\\-Saharan\\sAfrica\\snot\\sclassified\\sas\\sfragile\\ssituations|'
    'Sub\\-Saharan Africa excluding South Africa and Nigeria|South Asia \\(IDA \\& IBRD\\)|'
    'Sub-Saharan Africa \\(IDA \\& IBRD countries\\)|Middle East \\& North Africa \\(IDA \\& IBRD countries\\)|'
    'Latin America \\& the Caribbean \\(IDA \\& IBRD countries\\)|Europe \\& Central Asia \\(IDA \\& IBRD countries\\)|'
    'East Asia \\& Pacific \\(IDA \\& IBRD countries\\)|Sub\\-Saharan Africa|Sub\\-Saharan Africa \\(excluding high income\\)|'
    'Resource rich Sub\\-Saharan Africa countries|IDA countries not classified as fragile situations\\, excluding Sub\\-Saharan Africa|'
    'Non\\-resource rich Sub\\-Saharan Africa countries|Middle East \\& North Africa \\(excluding high income\\)|'
    'Middle East \\(developing only\\)|Latin America \\& Caribbean \\(excluding high income\\)|'
    'IBRD\\, including blend|Heavily indebted poor countries \\(HIPC\\)|'
    'IDA countries classified as fragile situations\\, excluding Sub\\-Saharan Africa|'
    'Europe \\& Central Asia \\(excluding high income\\)|East Asia \\& Pacific \\(excluding high income\\)|'
    'Sub\\-Saharan Africa \\(IDA\\-eligible countries\\)|IDA countries in Sub\\-Saharan Africa classified as fragile situations|'
    'South Asia \\(IDA\\-eligible countries\\)|IDA countries in Sub\\-Saharan Africa not classified as fragile situations|'
    'Middle East \\& North Africa \\(IDA\\-eligible countries\\)|Latin America \\& the Caribbean \\(IDA\\-eligible countries\\)|'
    'Europe \\& Central Asia \\(IDA\\-eligible countries\\)|East Asia \\& Pacific \\(IDA\\-eligible countries\\)|'
    'South Asia \\(IFC classification\\)|Middle East and North Africa \\(IFC classification\\)|'
    'Latin America and the Caribbean \\(IFC classification\\)|Europe and Central Asia \\(IFC classification\\)|'
    'East Asia and the Pacific \\(IFC classification\\)|Sub\\-Saharan Africa \\(IFC classification\\)|'
    'Sub\\-Saharan Africa \\(IBRD\\-only countries\\)|Latin America \\& the Caribbean \\(IBRD\\-only countries\\)|'
    'Middle East \\& North Africa \\(IBRD\\-only countries\\)|East Asia \\& Pacific \\(IBRD\\-only countries\\)|'
    'IBRD countries classified as high income|Europe \\& Central Asia \\(IBRD\\-only countries\\)|'
    'Sub\\-Saharan Africa \\(IDA \\& IBRD countries\\)|Sub-Saharan Africa \\(excluding high income\\)|'
    'Sub\\-Saharan Africa|South Asia \\(IDA \\& IBRD\\)|South Asia|Small states|Pre\\-demographic dividend|'
    'Post\\-demographic dividend|Pacific island small states|Other small states|OECD members|Not classified|'
    'North America|Middle income|Middle East \\& North Africa \\(IDA \\ & IBRD countries\\)|'
    'Middle East \\& North Africa \\(excluding high income\\)|Middle East \\& North Africa|'
    'Lower middle income|Low income|Low \\& middle income|Least developed countries\\: UN classification|'
    'Latin America \\& the Caribbean \\(IDA \\& IBRD countries\\)|Latin America \\& Caribbean \\(excluding high income\\)|'
    'Latin America \\& Caribbean|Late\\-demographic dividend|IDA total|IDA only|IDA blend|'
    'IDA \\& IBRD total|IBRD only|High income|Heavily indebted poor countries \\(HIPC\\)|'
    'Fragile and conflict affected situations|European Union|Europe \\& Central Asia \\(IDA \\& IBRD countries\\)|'
    'Europe \\& Central Asia \\(excluding high income\\)|Europe \\& Central Asia|Euro area|'
    'East Asia \\& Pacific \\(IDA \\& IBRD countries\\)|East Asia \\& Pacific \\(excluding high income\\)|'
    'East Asia \\& Pacific|Early\\-demographic dividend|Central Europe and the Baltics|Caribbean small states|'
    'Arab World|Africa Western and Central|Africa Eastern and Southern|'
    'Middle East, North Africa, Afghanistan \\& Pakistan \\(excluding high income\\)|'
    'Middle East, North Africa, Afghanistan \\& Pakistan \\(IDA \\& IBRD\\)|'
    'Middle East, North Africa, Afghanistan \\& Pakistan'
    ')$).*$'
)

# Retrieve country information from the World Bank API using the regex filter
countries_data = wbdata.get_countries(query=expression)
country_codes = [item["id"] for item in countries_data if item["id"] not in ('DNS', 'DSF')]

# Define the time range for data download
start_year = 1980
end_year = 2023

# Download the data (kept only in memory)
world_bank_data = download_world_bank_data(indicators, country_codes, start_year, end_year)

# Preprocessing data for time series analysis
world_bank_data['year'] = pd.to_datetime(world_bank_data['date']).dt.year  # Convert 'date' to 'Year' (extracting only the year)

# Drop column 'date'
world_bank_data = world_bank_data.drop(columns=['date'])

# Sort values for correct processing
world_bank_data = world_bank_data.sort_values(by=['country', 'year']).reset_index(drop=True)

# Convert object dtype columns to proper types
world_bank_data = world_bank_data.infer_objects()

# Identify numeric columns
#numeric_cols = world_bank_data.select_dtypes(include=[np.number]).columns.drop("year", errors="ignore")
numeric_cols = [
    "GDPpc_2017$",
    "Population_total",
    "Life_expectancy",
    "Literacy_rate",
    "Unemploymlent_rate",
    "Fertility_rate",
    "Poverty_ratio",
    "Primary_school_enrolmet_rate",
    "Energy_use",
    "Exports_2017$",
]

# Fill missing values with interpolation and forward-backward fill (only for numeric columns)
world_bank_data_temp = world_bank_data.groupby("country")[numeric_cols].apply(lambda group: group.interpolate(method="linear"))
world_bank_data_temp = world_bank_data_temp.ffill().bfill()
world_bank_data_temp = world_bank_data_temp.reset_index()

# Merge back non-numeric columns
non_numeric_cols = world_bank_data.drop(columns=numeric_cols).reset_index(drop=True)
world_bank_data_temp = pd.concat([non_numeric_cols, world_bank_data_temp[numeric_cols]], axis=1)
world_bank_data = world_bank_data_temp

# Save cleaned data to CSV
output_path = "world_bank_data_clean.csv"
world_bank_data.to_csv(output_path, index=False)

print(
    f"Data has been cleaned (missing values handled and type mismatches fixed) "
    f"and saved to '{output_path}'."
)
print(f"Final dataset shape: {world_bank_data.shape[0]} rows × {world_bank_data.shape[1]} columns.")