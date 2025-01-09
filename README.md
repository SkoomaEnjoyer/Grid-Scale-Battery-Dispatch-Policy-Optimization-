import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from hijri_converter import convert
from datetime import datetime
import calendar
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from scipy.stats import (
    norm, lognorm, expon, weibull_min, gamma, beta, chi2, cauchy, laplace, uniform,
    pareto, logistic, rayleigh, gumbel_l, gumbel_r, t, f, powerlaw, triang, genextreme,
    genpareto, levy, fisk, nakagami, vonmises, truncnorm
)
from pmdarima.arima import ndiffs
from statsmodels.tsa.arima.model import ARIMA
from fitter import Fitter
from datetime import datetime, timedelta
from scipy.stats import distributions
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.cm as cm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
idefrom sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import json
import copy
import matplotlib.dates as mdates

# Load the Excel file
file_path = r'C:\Users\ASUS\Desktop\Wind_data_dolar.xlsx'
wind_data = pd.read_excel(file_path)

# Creating a new dataframe 'analysis_dataframe' with only the required columns
analysis_dataframe = wind_data[['Year', 'Month', 'Day', 'Date', 'Hour', 
                                'MEP (TL/MWh)', 'Net Gen. (MWh)']]
cap_price = 2000
analysis_dataframe = analysis_dataframe.copy()
#Get Seasons
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
#Get Hours
hours = [hour for hour in range(24)]
years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
# Get weekday names from the calendar module
weekdays = [calendar.day_name[i] for i in range(7)]
# Define fixed Gregorian holidays (National Holidays)
gregorian_holidays = [
    '01-01',  # New Year's Day
    '04-23',  # National Sovereignty and Children’s Day
    '05-19',  # Commemoration of Atatürk, Youth and Sports Day
    '08-30',  # Victory Day
    '10-29',   # Republic Day
    '10-28',   # Half-day holiday for republic day
    '07-15',   # 15 Temmuz
    '04-01',   # İşçi Bayramı
]

# Define Hijri holidays for each relevant year for Ramazan Bayramı and Kurban Bayramı
hijri_holidays = {
    "Ramazan Bayramı Arefesi": [(9, 29)],  # Ramazan Bayramı Arefesi (last day of Ramadan)
    "Ramazan Bayramı": [(10, 1), (10, 2), (10, 3)],  # Ramazan Bayramı for 3 days starting from 1 Shawwal
    "Kurban Bayramı Arefesi": [(12, 9)],  # Kurban Bayramı Arefesi (day before 10 Dhu al-Hijjah)
    "Kurban Bayramı": [(12, 10), (12, 11), (12, 12), (12, 13)]  # Kurban Bayramı for 4 days starting from 10 Dhu al-Hijjah}
}

weekday_dict = {
     year: {season: {hour: [] for hour in hours} for season in seasons} for year in years
}
 
weekend_dict = {
     year: {season: {hour: [] for hour in hours} for season in seasons} for year in years
}

# List of discrete distributions to test
distributions = [
    stats.norm, stats.lognorm, stats.expon, stats.weibull_min, stats.gamma,
    stats.beta, stats.chi2, stats.cauchy, stats.laplace, stats.uniform,
    stats.pareto, stats.logistic, stats.rayleigh, stats.gumbel_l, stats.gumbel_r,
    stats.t, stats.f, stats.powerlaw, stats.triang, stats.genextreme,
    stats.genpareto, stats.levy, stats.fisk, stats.nakagami, stats.vonmises,
    stats.truncnorm
]

# Use seaborn's color palette for seasons
season_palette = sns.color_palette("coolwarm", n_colors=4)
season_color_map = dict(zip(["Winter", "Spring", "Summer", "Fall"], season_palette))

# Use seaborn's color palette for weekdays
weekday_palette = sns.color_palette("Accent", n_colors=7)
weekday_color_map = dict(zip(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], weekday_palette))
# Define the correct order for weekdays
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Define the season order
season_order = ["Fall", "Winter", "Spring", "Summer"]
###############################################################################
#############Functions that we will use for analyzing our data#################
# Function to get Hijri holidays in Gregorian format for every year between the specified range
def get_hijri_holidays_between_years(start_year, end_year, gregorian_holidays, hijri_holidays):
    hijri_to_gregorian_holidays = []
    
    # Loop through each year in the range
    for year in range(start_year, end_year + 1):
        # Calculate the corresponding Hijri year using the converter
        corresponding_hijri_year = convert.Gregorian(year, 1, 1).to_hijri().year
        
        # Full Holidays and Arefe days
        for bayram, dates in hijri_holidays.items():
            for hijri_month, hijri_day in dates:
                # Convert Hijri date to Gregorian date
                hijri_date = convert.Hijri(corresponding_hijri_year, hijri_month, hijri_day)
                gregorian_date = hijri_date.to_gregorian()
                
                # Assign labels: "half_day" for Arefe, "holiday" for others
                if "Arefesi" in bayram:
                    hijri_to_gregorian_holidays.append((gregorian_date.strftime('%Y-%m-%d'), "half_day"))  # Arefe is half holiday
                else:
                    hijri_to_gregorian_holidays.append((gregorian_date.strftime('%Y-%m-%d'), "holiday"))  # Full holiday
    # Convert list to dictionary for easier lookup
         
    return hijri_to_gregorian_holidays

def get_season(date):
    # Extract the month and day from the datetime object
    month = date.month
    day = date.day
    
    # Define the conditions for each season
    if (month == 12 and day >= 21) or (month == 1 or month == 2) or (month == 3 and day <= 19):
       return 'Winter'
    elif (month == 3 and day >= 20) or (month == 4 or month == 5) or (month == 6 and day <= 20):
        return 'Spring'
    elif (month == 6 and day >= 21) or (month == 7 or month == 8) or (month == 9 and day <= 21):
        return 'Summer'
    elif (month == 9 and day >= 22) or (month == 10 or month == 11) or (month == 12 and day <= 20):
        return 'Fall'
    
def prepare_the_dataframe(analysis_dataframe, hijri_to_gregorian_holidays, start_year, end_year, gregorian_holidays, hijri_holidays):
    # Correcting the issue by converting the 'Hour' to integer format and then adding it to the datetime object
    analysis_dataframe.loc[:, 'Hour'] = analysis_dataframe['Hour'].str[1:].astype(int)

    # Creating the DateTime object
    analysis_dataframe.loc[:, 'DateTime'] = pd.to_datetime(analysis_dataframe['Date']) + pd.to_timedelta(analysis_dataframe['Hour'], unit='h')

    # Dropping the now redundant columns
    analysis_dataframe = analysis_dataframe.drop(columns=['Year', 'Month', 'Day', 'Date', 'Hour'])

    # Reordering the columns to place 'DateTime' on the left
    analysis_dataframe = analysis_dataframe[['DateTime', 'MEP (TL/MWh)', 'Net Gen. (MWh)']]
    analysis_dataframe['EventStamp'] = 'Weekday'  # Default to 'Weekday'
    # Assign 'Weekend' to rows where the day of the week is Saturday (5) or Sunday (6)
    analysis_dataframe.loc[analysis_dataframe['DateTime'].dt.dayofweek >= 5, 'EventStamp'] = 'Weekend'
    
    # Extract start and end years from the dataset's 'DateTime' column
    start_year = analysis_dataframe['DateTime'].dt.year.min()  # Minimum year in the dataset
    end_year = analysis_dataframe['DateTime'].dt.year.max()    # Maximum year in the dataset
    hijri_to_gregorian_holidays = get_hijri_holidays_between_years(start_year, end_year, gregorian_holidays, hijri_holidays)
    
    # Loop through each tuple in the holiday list
    for holiday_tuple in hijri_to_gregorian_holidays:
        holiday_date_str = holiday_tuple[0]  # The first element (string date)
        day_type = holiday_tuple[1]          # The second element (holiday type)
        
        # Convert the string date to a datetime object
        holiday_date = datetime.strptime(holiday_date_str, '%Y-%m-%d')
    
        # Find the rows in 'analysis_dataframe' where the 'DateTime' column matches the holiday date
        # We use `.dt.date` to ignore time and only match the date
        matching_rows = analysis_dataframe[analysis_dataframe['DateTime'].dt.date == holiday_date.date()]
        
        # If any rows are found, update the 'EventStamp' column for these rows
        if matching_rows.empty == False:
            # Update the 'EventStamp' for the matching rows
            analysis_dataframe.loc[analysis_dataframe['DateTime'].dt.date == holiday_date.date(), 'EventStamp'] = day_type
    #Stamp the seasons
    analysis_dataframe['Season'] = analysis_dataframe['DateTime'].apply(get_season)
    #Create a new column that captures the day of the week (Monday = 0, Sunday = 6)
    analysis_dataframe['DayOfWeek'] = analysis_dataframe['DateTime'].dt.day_name()
    # Filter out rows where 'EventStamp' is either 'holiday' or 'half_day'
    ##THIS IS WHERE WE EXCLUDE HOLIDAYS##
    filtered_analysis_dataframe = analysis_dataframe[~analysis_dataframe['EventStamp'].isin(['holiday', 'half_day'])]
    # Create a dataframe with only 'holiday' and 'half_day' values
    holiday_halfday_dataframe = analysis_dataframe[analysis_dataframe['EventStamp'].isin(['holiday', 'half_day'])]
    # Add a 'Year' column for easier grouping
    holiday_halfday_dataframe.loc[:, 'Year'] = holiday_halfday_dataframe['DateTime'].dt.year
    filtered_analysis_dataframe.loc[:, 'Year'] = filtered_analysis_dataframe['DateTime'].dt.year
    analysis_dataframe.loc[:, 'Year'] = analysis_dataframe['DateTime'].dt.year
    return analysis_dataframe, filtered_analysis_dataframe, holiday_halfday_dataframe

    # Extract relevant columns and make a copy to avoid SettingWithCopyWarning
    wind_df = analysis_dataframe[['DateTime', 'Net Gen. (MWh)']].copy()
    
    # Ensure DateTime is set as the index and in datetime format
    wind_df['DateTime'] = pd.to_datetime(wind_df['DateTime'])
    wind_df.set_index('DateTime', inplace=True)

    # Handle any negative values if necessary
    wind_df['Net Gen. (MWh)'] = wind_df['Net Gen. (MWh)'].apply(lambda x: max(x, 0))
    
    # Fit an ARIMA model (simple order)
    model = ARIMA(wind_df['Net Gen. (MWh)'], order=(2, 0, 2))  # You can adjust (p, d, q) as needed
    model_fit = model.fit()
    
    # Forecast for the next year (8760 hours)
    forecast = model_fit.forecast(steps=8760)
    
    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame({
        'DateTime': pd.date_range(start=wind_df.index[-1] + pd.Timedelta(hours=1), periods=8760, freq='H'),
        'Forecasted Net Gen. (MWh)': forecast
    })
    
    # Plot actual data and forecast
    plt.figure(figsize=(14, 7))
    plt.plot(wind_df.index[-24 * 7:], wind_df['Net Gen. (MWh)'].iloc[-24 * 7:], label='Actuals')
    plt.plot(forecast_df['DateTime'][:24 * 7], forecast_df['Forecasted Net Gen. (MWh)'][:24 * 7], label='Forecast (First Week)', color='orange')
    plt.legend()
    plt.xlabel('DateTime')
    plt.ylabel('Net Gen. (MWh)')
    plt.title('Hourly Net Generation Forecast (ARIMA)')
    plt.grid(True)
    plt.show()
    
    
    forecast_df.to_excel(r"C:\Users\ASUS\Desktop\forecast.xlsx", index=False, sheet_name='Sheet1')
    return forecast_df

def choose_the_day_and_year_to_analyze (filtered_analysis_dataframe, day, year):
    
    # Filter the dataframe to get only the chosen day (DayOfWeek == daynum) of the year we have chosen
    day_year_df = filtered_analysis_dataframe[
        (filtered_analysis_dataframe['DateTime'].dt.year == year) & 
        (filtered_analysis_dataframe['DayOfWeek'] == day)
        ]

    # Selecting only the relevant columns: DateTime, MEP, Net Gen.
    day_year_df = day_year_df[['DateTime', 'MEP (TL/MWh)', 'EventStamp', 'Season', 'DayOfWeek']]
    day_year_df = day_year_df.reset_index(drop = True)
    
    # Create an empty dictionary to store subsets
    season_hourly_subsets_dict = {}
    
    # List of unique seasons in the data
    seasons = day_year_df['Season'].unique()
    
    # Loop through each season
    for season in seasons:
        # Filter the dataframe for the current season
        season_df = day_year_df[day_year_df['Season'] == season]
        # Create a dictionary to store hourly subsets for this season
        hourly_dict = {}
        
        # Loop through each hour (0 to 23)
        for hour in range(24):
            # Filter rows where the 'DateTime' hour equals the current hour
            hourly_subset = season_df[season_df['DateTime'].dt.hour == hour]
            
            # Add to the hourly dictionary
            hourly_dict[hour] = hourly_subset
        season_hourly_subsets_dict[season] = hourly_dict
    
    return day_year_df, season_hourly_subsets_dict

def create_violin_graph_hours(start_year, end_year, season_colors, day_year_df, season_hourly_subsets_dict, seasons, weekdays, everyyear_everyday_season_hourly_subsets_dict):
    # Dictionary to store the calculated values
    box_plot_stats = {}
    
    for year in range(start_year, end_year + 1):  # Adjust year range based on your dataset
        for day in weekdays:
            
            # Create a list to store data for plotting
            plot_data = []
            
            # Iterate through seasons and hours
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                for hour in range(24):
                    # Extract the hourly data from the dictionary
                    hourly_data = everyyear_everyday_season_hourly_subsets_dict[year][day][season][hour]['MEP (TL/MWh)']
                    
                    # Append data for each hour, day, and season
                    for value in hourly_data:
                        plot_data.append([day, season, hour, value])
    
            # Convert the list into a pandas DataFrame for Seaborn plotting
            df = pd.DataFrame(plot_data, columns=['Day', 'Season', 'Hour', 'Cost'])
    
            # Plot using Seaborn
            plt.figure(figsize=(30, 20))
    
            # Plot violin plots with specific colors
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                # Plot the violin plot first (semi-transparent to allow box plot visibility)
                sns.violinplot(x='Hour', y='Cost', data=df[df['Season'] == season], color=season_colors[season], alpha=0.6)
    
                # Plot the box plot on top of the violin plot
                sns.boxplot(x='Hour', y='Cost', data=df[df['Season'] == season], width=0.2, color='black', showcaps=False, 
                            boxprops={'facecolor':'None'}, whiskerprops={'linewidth':2}, zorder=2)
    
                # Store the box plot stats in the dictionary
                for hour in range(24):
                    # Extract the data for this hour and season
                    data = df[(df['Hour'] == hour) & (df['Season'] == season)]['Cost']
                    
                    if len(data) > 0:  # Only proceed if there's data for this hour
                        # Calculate quartiles and median
                        q1 = np.percentile(data, 25)
                        median = np.median(data)
                        q3 = np.percentile(data, 75)
                        min_value = np.min(data)  # Calculate the minimum price
                        max_value = np.max(data)  # Calculate the maximum price
                        
                        # Create a key based on year, day, season, and hour
                        key = (year, day, season, hour)
                        
                        # Store the calculated values in the dictionary
                        box_plot_stats[key] = {
                            'Q1': q1, 
                            'Median': median, 
                            'Q3': q3, 
                            'Min': min_value,   # Store min value
                            'Max': max_value    # Store max value
                        }
    
            # Add title and labels
            plt.title(f'Hourly Electricity Price Distribution for {day} in {year}')
            plt.xlabel('Hour of Day')
            plt.ylabel('Electricity Price')
            
            # Show the plot
            plt.show()
    return box_plot_stats, plt.show()

def create_heatmap_hours(start_year, end_year, day_year_df, season_hourly_subsets_dict, seasons, weekdays, everyyear_everyday_season_hourly_subsets_dict):
    ##Heatmap
    for year in range(start_year, end_year + 1):  # Adjust year range based on your dataset
        for day in weekdays:
            
            # Create a list to store data for plotting
            plot_data = []
    
            # Iterate through seasons and hours
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                for hour in range(24):
                    # Extract the hourly data from the dictionary
                    hourly_data = everyyear_everyday_season_hourly_subsets_dict[year][day][season][hour]['MEP (TL/MWh)']
                    # Append data for each hour, day, and season
                    for value in hourly_data:
                        plot_data.append([hour, season, value])
    
            # Convert the list into a pandas DataFrame for Seaborn plotting
            df = pd.DataFrame(plot_data, columns=['Hour', 'Season', 'Cost'])
    
            # Create a pivot table to prepare data for the heatmap (average electricity price per hour and season)
            heatmap_data = df.pivot_table(index='Hour', columns='Season', values='Cost', aggfunc='mean')
    
            # Plot the heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
            # Add title and labels
            plt.title(f'Average Hourly Electricity Price Heatmap for {day} in {year}')
            plt.xlabel('Season')
            plt.ylabel('Hour of Day')
    
            # Show the plot
            plt.show()
            

    return plt.show()


def create_violin_graph_years(start_year, end_year, day_year_df, season_hourly_subsets_dict, seasons, weekdays, everyyear_everyday_season_hourly_subsets_dict):
    ##Days in x axis, 52 average price for every single weekday and year
    # Iterate through each year to generate a violin plot
    # Dictionary to store the max and min values for each year and day
    max_min_values = {}
    
    # Iterate through each year to generate a violin plot
    for year in range(start_year, end_year + 1):  # Adjust year range based on your dataset
        
        # Create a list to store data for plotting
        plot_data = []
    
        # Initialize a dictionary for storing max and min values for the current year
        yearly_max_min = {}
    
        # Loop through each day of the week
        for day in weekdays:
            
            # Collect daily data for the entire day (all hours across seasons)
            daily_data = []
    
            for season in seasons:
                for hour in range(24):
                    try:
                        # Extract the hourly data for the specific day
                        hourly_data = everyyear_everyday_season_hourly_subsets_dict[year][day][season][hour]['MEP (TL/MWh)']
                        
                        # Store the day, price, and season for each hour
                        daily_data.extend(hourly_data)
                        
                        for price in hourly_data:
                            plot_data.append([day, season, price])  # Store day, season, and individual price
                    except KeyError:
                        # If data is missing, skip this hour
                        continue
            
            # Calculate max and min values for the current day if data is available
            if len(daily_data) > 0:
                day_max = max(daily_data)
                day_min = min(daily_data)
                # Store max and min values for the day in the yearly dictionary
                yearly_max_min[day] = {'Max': day_max, 'Min': day_min}
    
        # Store the yearly max and min values
        max_min_values[year] = yearly_max_min
    
        # Convert the list into a pandas DataFrame for Seaborn plotting
        df = pd.DataFrame(plot_data, columns=['Day', 'Season', 'Price'])
    
        # Plot using Seaborn with seasons as the hue
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='Day', y='Price', hue='Season', data=df, palette='Set2', linewidth=1)
    
        # Add title and labels
        plt.title(f'Electricity Price Distribution by Day of Week and Season for {year}')
        plt.xlabel('Day of the Week')
        plt.ylabel('Electricity Price')
    
        # Show the plot
        plt.show()
    return plt.show()


def remove_cap_values(dataframe, cap_price):
    # Calculate the percentage of data points >= cap price for each year
    yearly_cap_percentage = []
    for year, group in dataframe.groupby('Year'):
        total_count = len(group)
        cap_count = (group['MEP (TL/MWh)'] >= (cap_price * 0.95)).sum()
        cap_percentage = (cap_count / total_count) * 100
        yearly_cap_percentage.append({'Year': year, 'Cap_Percentage': cap_percentage})

    # Convert the list of dictionaries to a pandas DataFrame
    yearly_cap_percentage = pd.DataFrame(yearly_cap_percentage)

    # Identify and list years where the percentage of cap prices is 20% or more
    years_to_discard = yearly_cap_percentage[yearly_cap_percentage['Cap_Percentage'] >= 5]['Year'].tolist()

    # Filter the main dataframe to exclude rows corresponding to the years to discard
    if years_to_discard:
        dataframe = dataframe[~dataframe['Year'].isin(years_to_discard)]
        
    for i in range(len(yearly_cap_percentage)):
        year = yearly_cap_percentage.loc[i, 'Year']
        percentage = yearly_cap_percentage.loc[i, 'Cap_Percentage']
    
        # Check if the percentage exceeds the threshold
        if percentage > 0:
            # Remove rows for the year where MEP (TL/MWh) exceeds the cap value
            dataframe = dataframe[
                ~((dataframe['Year'] == year) & 
                  (dataframe['MEP (TL/MWh)'] >= (cap_price * 0.95)))
                ]
            
    return yearly_cap_percentage, dataframe

def convert_dataframe_to_dictionary(dataframe):
    # Iterate over the filtered dataframe to populate the dictionaries
    for _, row in dataframe.iterrows():
        # Extract the day of the week, season, hour, year, and value information
        day_of_week = row['DateTime'].dayofweek  # Monday = 0, Sunday = 6
        season = row['Season']
        hour = row['DateTime'].hour
        year = str(row['DateTime'].year)  # Convert year to string to match dictionary keys
        value = row['MEP (TL/MWh)']
        datetime_obj = row['DateTime']  # Get the datetime object
        
        # Append the (value, datetime_obj) tuple to the appropriate dictionary based on weekday/weekend
        if day_of_week < 5:  # Weekday
            # Append to weekday_dict[year][season][hour]
            weekday_dict[year][season][hour].append((value, datetime_obj))
        else:  # Weekend
            # Append to weekend_dict[year][season][hour]
            weekend_dict[year][season][hour].append((value, datetime_obj))
    return weekday_dict, weekend_dict

def remove_and_store_the_extreme_points(dictionary, seasons, hours):
    years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    # List to store all extreme points with their respective keys
    extreme_points = []

    # Iterate over all seasons and hours
    for year in years:
        for season in seasons:  # Ensure you process each season individually
            for hour in hours:  # Process each hour

                # Extract data for the current year, season, and hour
                data_with_datetime = np.array(dictionary[year][season][hour])
                
                # Extract data values and datetime objects from tuples
                data_values = np.array([item[0] for item in data_with_datetime])  # Extract values
                datetime_values = np.array([item[1] for item in data_with_datetime])  # Extract datetime objects
                
                # Calculate the median
                median = np.median(data_values)
                
                # Calculate the absolute deviations from the median
                absolute_deviations = np.abs(data_values - median)
                
                # Calculate the MAD (Median Absolute Deviation)
                mad = np.median(absolute_deviations) if np.median(absolute_deviations) != 0 else 1  # Avoid division by zero

                # Calculate the modified Z-scores
                modified_z_scores = 0.6745 * (data_values - median) / mad

                # Define a threshold for extreme points (e.g., |modified z| > 10.0)
                threshold = 10  # Use a stricter threshold for extreme points
                current_extreme_points = [
                    (value, dt) for value, dt, mz in zip(data_values, datetime_values, modified_z_scores)
                    if abs(mz) > threshold
                ]

                # Save the extreme points to the main list with their key information
                if current_extreme_points:
                    extreme_points.extend(current_extreme_points)

                # Remove extreme points from the original array in the dictionary
                cleaned_data_with_datetime = [
                    (value, dt) for value, dt, mz in zip(data_values, datetime_values, modified_z_scores)
                    if abs(mz) <= threshold
                ]

                # Update the cleaned data for this hour
                dictionary[year][season][hour] = cleaned_data_with_datetime
            
    return dictionary, extreme_points

def convert_to_integers(dictionary):
    for year in dictionary:  # Iterate over years
        for season in dictionary[year]:  # Iterate over seasons
            for hour in dictionary[year][season]:  # Iterate over hours
                # Iterate over each tuple (value, datetime_obj)
                dictionary[year][season][hour] = [
                    (int(value), datetime_obj)  # Convert value to integer
                    for value, datetime_obj in dictionary[year][season][hour]
                ]
    return dictionary

def aggregate_dictionary(dataframe):
  # Initialize the aggregated dictionary with seasons and 24-hour keys
    aggregated_dict = {season: {hour: [] for hour in range(24)} for season in ["Winter", "Spring", "Summer", "Fall"]}

    for year, seasons in dataframe.items():
        for season, hours in seasons.items():
            # Ensure the season exists in the aggregated dictionary
            if season not in aggregated_dict:
                aggregated_dict[season] = {hour: [] for hour in range(24)}
            
            for hour, values in hours.items():
                # Ensure the hour is valid and aggregate values
                if isinstance(hour, int) and 0 <= hour < 24:
                    aggregated_dict[season][hour].extend(values)

    return aggregated_dict

def find_the_distributions(dictionary, distributions, seasons):
    # List to store the best-fitting distribution for each key
    best_fitting_distributions = []
    # Iterate over all seasons and hours dynamically
    for season in dictionary.keys():  # Iterate over seasons in the dictionary
        for hour in range(24):  # Iterate over 24 hours
            # Extract data for the current season and hour
            data_with_datetime = np.array(dictionary[season][hour])
            
            # Extract data values and datetime values from tuples
            data_values = np.array([item[0] for item in data_with_datetime])  # Values for distribution fitting
            datetime_values = np.array([item[1] for item in data_with_datetime])  # Corresponding datetime values
            
            # Initialize variables for tracking the best fit
            best_fit_name = None
            best_p_value = 0  # Initialize with a low p-value to start comparison
            
            # Fit and test each distribution
            for dist in distributions:
                dist_name = dist.name if hasattr(dist, 'name') else str(dist)
                
                # Fit the distribution to the data
                params = dist.fit(data_values)
                
                # Generate the expected frequencies based on the fitted distribution
                observed_values, bin_edges = np.histogram(data_values, bins='auto', density=False)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                expected_values = dist.pdf(bin_centers, *params) * len(data_values) * np.diff(bin_edges)
                
                # Normalize the expected frequencies to match the sum of the observed frequencies
                expected_values *= observed_values.sum() / expected_values.sum()
                
                # Perform the chi-squared test
                chi2, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
                
                # Check if this distribution has the best p-value so far
                if p_value > best_p_value:
                    best_p_value = p_value
                    best_fit_name = dist_name
            
            # Append the best-fitting distribution for this season and hour
            best_fitting_distributions.append((f"{season} - Hour {hour}", best_fit_name, best_p_value))
    
    # Output the best-fitting distributions
    #for key, dist_name, p_value in best_fitting_distributions:
        #print(f"{key}: Best fit = {dist_name} (p-value = {p_value:.4f})")

                    
    return best_fitting_distributions

def analyze_and_visualize_the_market_clearing_price (analysis_dataframe, filtered_analysis_dataframe, seasons, weekdays, weekday_dict, weekend_dict, distributions, cap_price, holiday_halfday_dataframe, hours):
   
    #Find out the time interval which out data is in
    start_year = analysis_dataframe['DateTime'].dt.year.min()  # Minimum year in the dataset
    end_year = analysis_dataframe['DateTime'].dt.year.max()    # Maximum year in the dataset
    # Define a color palette for the seasons
    season_colors = {'Winter': 'blue', 'Spring': '#006400', 'Summer': 'yellow', 'Fall': 'brown'}
    #Store all the hours, days, seasons, and years in a single dictionary
    everyyear_everyday_season_hourly_subsets_dict = {}
    for year in range(start_year, end_year + 1):
        everyday_season_hourly_subsets_dict = {}
        for day in weekdays:
            day_year_df, season_hourly_subsets_dict = choose_the_day_and_year_to_analyze(filtered_analysis_dataframe, day, year)
            everyday_season_hourly_subsets_dict[day] = season_hourly_subsets_dict 
        everyyear_everyday_season_hourly_subsets_dict[year] = everyday_season_hourly_subsets_dict
    box_plot_stats, x = create_violin_graph_hours(start_year, end_year, season_colors, day_year_df, season_hourly_subsets_dict, seasons, weekdays, everyyear_everyday_season_hourly_subsets_dict)
    create_heatmap_hours(start_year, end_year, day_year_df, season_hourly_subsets_dict, seasons, weekdays, everyyear_everyday_season_hourly_subsets_dict)
    create_violin_graph_years(start_year, end_year, day_year_df, season_hourly_subsets_dict, seasons, weekdays, everyyear_everyday_season_hourly_subsets_dict) 

    ####Identify and list years where the percentage of cap prices is 5% or more###
    (yearly_cap_percentage_analysis, analysis_dataframe) = remove_cap_values(analysis_dataframe, cap_price)
    (yearly_cap_percentage_filtered_analysis, filtered_analysis_dataframe) = remove_cap_values(filtered_analysis_dataframe, cap_price)
    (yearly_cap_percentage_holdiay_halfday, holiday_halfday_dataframe) = remove_cap_values(holiday_halfday_dataframe, cap_price)
    #####there are no years to discard nor values###
    ##########################################################
    ####Store the data in dictionaries####
    (weekday_dict_filtered_data, weekend_dict_filtered_data) = convert_dataframe_to_dictionary(filtered_analysis_dataframe)
    (weekday_dict_holiday, weekend_dict_holiday) = convert_dataframe_to_dictionary(holiday_halfday_dataframe)    
    ####################################
    ###Extract and Store the Extreme Points
    (weekday_dict_filtered_data, extreme_points_weekday_filtered) = remove_and_store_the_extreme_points(weekday_dict_filtered_data, seasons, hours)
    (weekend_dict_filtered_data, extreme_points_weekend_filtered) = remove_and_store_the_extreme_points(weekend_dict_filtered_data, seasons, hours)
    (weekday_dict_holiday, extreme_points_weekday_holiday) = remove_and_store_the_extreme_points(weekday_dict_holiday, seasons, hours)
    (weekend_dict_holiday, extreme_points_weekend_holiday) = remove_and_store_the_extreme_points(weekend_dict_holiday, seasons, hours)
    #####################################
    ####Lets convert the floats to integers###
    weekday_dict_filtered_data = convert_to_integers(weekday_dict_filtered_data)
    weekend_dict_filtered_data = convert_to_integers(weekend_dict_filtered_data)
    weekday_dict_holiday = convert_to_integers(weekday_dict_holiday)
    weekend_dict_holiday = convert_to_integers(weekend_dict_holiday)
    #####################################
    #####Aggregate Dictionary#####
    weekday_dict_filtered_data = aggregate_dictionary(weekday_dict_filtered_data)
    weekend_dict_filtered_data = aggregate_dictionary(weekend_dict_filtered_data)
    weekday_dict_holiday = aggregate_dictionary(weekday_dict_holiday)
    weekend_dict_holiday = aggregate_dictionary(weekend_dict_holiday)
    #######################################
    ####Lets find the distributions#####
    weekday_dict_filtered_data_distributions = find_the_distributions(weekday_dict_filtered_data, distributions, seasons)
    weekend_dict_filtered_data_distributions = find_the_distributions(weekend_dict_filtered_data, distributions, seasons)
    weekday_dict_holiday_distributions = find_the_distributions(weekday_dict_holiday, distributions, seasons)
    weekend_dict_holiday_distributions = find_the_distributions(weekend_dict_holiday, distributions, seasons)
    ####################################
       
    return everyyear_everyday_season_hourly_subsets_dict
###############################################################################
###############################################################################
#(analysis_dataframe, filtered_analysis_dataframe, holiday_halfday_dataframe) = prepare_the_dataframe(analysis_dataframe, hijri_to_gregorian_holidays, start_year, end_year, gregorian_holidays, hijri_holidays)
start_year = analysis_dataframe['Date'].dt.year.min()  # Minimum year in the dataset
end_year = analysis_dataframe['Date'].dt.year.max()    # Maximum year in the dataset
hijri_to_gregorian_holidays = get_hijri_holidays_between_years(start_year, end_year, gregorian_holidays, hijri_holidays)
(holiday_halfday_dataframe, filtered_analysis_dataframe, holiday_halfday_dataframe) = prepare_the_dataframe(analysis_dataframe, hijri_to_gregorian_holidays, start_year, end_year, gregorian_holidays, hijri_holidays)
analyze_and_visualize_the_market_clearing_price (analysis_dataframe, filtered_analysis_dataframe, seasons, weekdays, weekday_dict, weekend_dict, distributions, cap_price, holiday_halfday_dataframe, hours)

# Initialize a list to store extreme points detected by the IQR method
def detect_and_remove_extreme_points(dataframe, seasons, hours, iqr_multiplier=8):
    """
    Detect and remove extreme points using the IQR method.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing 'Year', 'Season', 'DateTime', and 'MEP (TL/MWh)' columns.
        seasons (list): List of seasons to filter (e.g., ['Winter', 'Spring', 'Summer', 'Fall']).
        hours (list): List of hours to filter (e.g., range(24)).
        iqr_multiplier (float): Multiplier for IQR to define extreme points (default is 8).

    Returns:
        tuple: A tuple containing:
            - cleaned_data (pd.DataFrame): DataFrame with extreme points removed.
            - extreme_points_df (pd.DataFrame): DataFrame containing detected extreme points.
    """
    extreme_points = []

    # Iterate over the unique years in the DataFrame
    for year in dataframe['Year'].unique():
        for season in seasons:  # Iterate through the defined seasons
            for hour in hours:  # Iterate through each hour (0-23)
                # Filter the DataFrame for the current year, season, and hour
                filtered_data = dataframe[
                    (dataframe['Year'] == year) &
                    (dataframe['Season'] == season) &
                    (dataframe['DateTime'].dt.hour == hour)
                ]

                # Proceed only if the filtered data is not empty
                if not filtered_data.empty:
                    # Extract the MEP values
                    data_values = filtered_data['MEP (TL/MWh)'].values

                    # Calculate the first and third quartiles
                    q1 = np.percentile(data_values, 25)
                    q3 = np.percentile(data_values, 75)
                    iqr = q3 - q1

                    # Define the lower and upper bounds for detecting outliers
                    lower_bound = q1 - iqr_multiplier * iqr
                    upper_bound = q3 + iqr_multiplier * iqr

                    # Extract extreme points
                    current_extreme_points = [
                        (value, dt) for value, dt in zip(data_values, filtered_data['DateTime'])
                        if value < lower_bound or value > upper_bound
                    ]

                    # Save extreme points to the main list
                    if current_extreme_points:
                        extreme_points.extend(current_extreme_points)

                    # Remove extreme points from the DataFrame
                    dataframe = dataframe[
                        ~((dataframe['Year'] == year) &
                          (dataframe['Season'] == season) &
                          (dataframe['DateTime'].dt.hour == hour) &
                          (dataframe['MEP (TL/MWh)'].isin([ep[0] for ep in current_extreme_points])))
                    ]

    # Convert extreme points to a DataFrame for analysis
    extreme_points_df = pd.DataFrame(extreme_points, columns=['MEP (TL/MWh)', 'DateTime'])

    return dataframe, extreme_points_df

(filtered_analysis_dataframe, extreme_points_df) = detect_and_remove_extreme_points(filtered_analysis_dataframe, seasons, hours, iqr_multiplier=8)

# Extract the year from the DateTime column and create a new column 'Year'
extreme_points_df['Year'] = extreme_points_df['DateTime'].dt.year

# Count the number of extreme values per year
extreme_values_per_year = extreme_points_df['Year'].value_counts().sort_index()

# Print the results
print("Extreme values per year:")
print(extreme_values_per_year)

###################################################################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Horizontal Aggregation: Every single hour's average price for every day and every season
# Extract the hour from the DateTime column
filtered_analysis_dataframe['Hour'] = filtered_analysis_dataframe['DateTime'].dt.hour

# Perform the aggregation
aggregated_horizontal_data = (
    filtered_analysis_dataframe
    .groupby(['Season', 'DayOfWeek', 'Hour'])
    .agg(
        Total_MEP=('MEP (TL/MWh)', 'sum'),
        Count=('MEP (TL/MWh)', 'count'),
        Average_MEP=('MEP (TL/MWh)', 'mean')
    )
    .reset_index()
)

# Convert 'DayOfWeek' and 'Season' columns to categorical with the defined orders
aggregated_horizontal_data['DayOfWeek'] = pd.Categorical(
    aggregated_horizontal_data['DayOfWeek'], categories=weekday_order, ordered=True
)

aggregated_horizontal_data['Season'] = pd.Categorical(
    aggregated_horizontal_data['Season'], categories=season_order, ordered=True
)

# Sort the data for clarity
aggregated_horizontal_hourly_data = aggregated_horizontal_data.sort_values(
    by=['Season', 'DayOfWeek', 'Hour']
).reset_index(drop=True)

########################################################
########################################################
# Vertical Aggregation: Every single day's average price
# Extract the date part from the DateTime column
filtered_analysis_dataframe['Date'] = filtered_analysis_dataframe['DateTime'].dt.date

# Group by 'Date' and calculate the daily aggregates
vertical_daily_aggregates = (
    filtered_analysis_dataframe.groupby('Date')['MEP (TL/MWh)']
    .agg(Average_MEP=('sum'), Count='count')  # Aggregate: Sum and Count
    .reset_index()  # Convert grouped data back to a DataFrame
)

# Calculate the average MEP for each day
vertical_daily_aggregates['Average MEP (TL/MWh)'] = vertical_daily_aggregates['Average_MEP'] / vertical_daily_aggregates['Count']

# Drop the count column if not needed (optional)
vertical_daily_aggregates = vertical_daily_aggregates[['Date', 'Average MEP (TL/MWh)']]
vertical_daily_aggregates.rename(columns={'Average MEP (TL/MWh)': 'Average_MEP'}, inplace=True)
##############################################################################
###############################################################################
################################aggregated_monthly_data###############################################

# Create 'YearMonth' column by extracting year and month from 'DateTime'
filtered_analysis_dataframe['YearMonth'] = filtered_analysis_dataframe['DateTime'].dt.to_period('M')

# Perform monthly aggregation
aggregated_monthly_data = (
    filtered_analysis_dataframe
    .groupby('YearMonth')
    .agg(
        Total_MEP=('MEP (TL/MWh)', 'sum'),
        Count=('MEP (TL/MWh)', 'count'),
        Average_MEP=('MEP (TL/MWh)', 'mean')  # Single average for the whole month
    )
    .reset_index()
)

# Optionally convert 'YearMonth' to datetime for better handling
aggregated_monthly_data['YearMonth'] = aggregated_monthly_data['YearMonth'].dt.to_timestamp()

###############################################################################
###############################################################################
#########################Plot the dataframes###################################
#################Plot the horizontal dataframe#################################
def plot_combined_weekday_season(data, weekday_color_map, weekday_order, metric_col='Average_MEP', season_col='Season', day_col='DayOfWeek', hour_col='Hour'):
    """
    Plot combined hourly averages for weekdays grouped by season.
    """
    plt.figure(figsize=(12, 6))
    seasons = data[season_col].unique()

    for season in seasons:
        season_data = data[data[season_col] == season]
        for weekday in weekday_order:  # Iterate in the correct order
            weekday_data = season_data[season_data[day_col] == weekday]
            if not weekday_data.empty:
                plt.plot(
                    weekday_data[hour_col],
                    weekday_data[metric_col],
                    label=f"{weekday} ({season})",
                    color=weekday_color_map.get(weekday, 'gray'),
                    marker='o',
                    linewidth=1,
                )
    plt.title(f'Hourly {metric_col} by Season and Weekday', fontsize=16)
    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel(f'{metric_col}', fontsize=12)
    plt.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_seasonal_transparency(data, weekday_color_map, weekday_order, season_order, metric_col='Average_MEP', season_col='Season', day_col='DayOfWeek', hour_col='Hour'):
    """
    Plot transparent lines for each weekday within a season, maintaining the order.
    """
    # Ensure the columns have categorical data types with specified order
    data[season_col] = pd.Categorical(data[season_col], categories=season_order, ordered=True)
    data[day_col] = pd.Categorical(data[day_col], categories=weekday_order, ordered=True)
    
    plt.figure(figsize=(12, 8))
    seasons = data[season_col].cat.categories  # Use ordered categories

    for season in seasons:
        plt.figure(figsize=(10, 6))
        season_data = data[data[season_col] == season]
        for weekday in weekday_order:  # Iterate in the correct order
            weekday_data = season_data[season_data[day_col] == weekday]
            if not weekday_data.empty:
                plt.plot(
                    weekday_data[hour_col],
                    weekday_data[metric_col],
                    label=f"{weekday}",
                    color=weekday_color_map.get(weekday, 'gray'),
                    alpha=0.5,  # Transparency
                    marker='o',
                    linewidth=1,
                )
        plt.title(f'Hourly {metric_col} for {season}', fontsize=16)
        plt.xlabel('Hour of the Day', fontsize=12)
        plt.ylabel(f'{metric_col}', fontsize=12)
        plt.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
        plt.grid()
        plt.tight_layout()
        plt.show()

def plot_individual_weekday(data, season_color_map, weekday_order, metric_col='Average_MEP', season_col='Season', day_col='DayOfWeek', hour_col='Hour'):
    """
    Plot individual weekdays across all seasons.
    """
    for weekday in weekday_order:  # Iterate in the correct order
        plt.figure(figsize=(10, 6))
        for season in data[season_col].unique():
            season_weekday_data = data[
                (data[season_col] == season) & (data[day_col] == weekday)
            ]
            if not season_weekday_data.empty:
                plt.plot(
                    season_weekday_data[hour_col],
                    season_weekday_data[metric_col],
                    label=f"{season}",
                    color=season_color_map.get(season, 'gray'),
                    marker='o',
                    linewidth=2,
                )
        plt.title(f'Hourly {metric_col} for {weekday}', fontsize=16)
        plt.xlabel('Hour of the Day', fontsize=12)
        plt.ylabel(f'{metric_col}', fontsize=12)
        plt.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
        plt.grid()
        plt.tight_layout()
        plt.show()


# Call the functions
plot_combined_weekday_season(
    data=aggregated_horizontal_hourly_data,
    weekday_color_map=weekday_color_map,
    metric_col='Average_MEP',
    weekday_order=weekday_order
)

# Call the updated function
plot_seasonal_transparency(
    data=aggregated_horizontal_hourly_data,
    weekday_color_map=weekday_color_map,
    metric_col='Average_MEP',
    weekday_order=weekday_order,
    season_order=season_order
)

plot_individual_weekday(
    data=aggregated_horizontal_hourly_data,
    season_color_map=season_color_map,
    metric_col='Average_MEP',
    weekday_order=weekday_order
)
###############################################################################
###############################################################################
# Plot 2: Vertical Daily Aggregates
# Generate a colormap based on the number of days
def plot_daily_aggregates(data, date_col, value_col, title="Daily Aggregates Over Time", xlabel="Date", ylabel="Value"):
    """
    Plots daily aggregates with each day represented by a different color.
    
    Parameters:
        data (DataFrame): The dataset containing the daily aggregates.
        date_col (str): The column name for dates.
        value_col (str): The column name for the values to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Generate a colormap based on the number of days
    num_days = len(data)
    colormap = cm.get_cmap('viridis', num_days)

    # Create the plot
    plt.figure(figsize=(20, 8))
    for i in range(num_days - 1):  # Loop through days and plot with different colors
        plt.plot(
            data[date_col].iloc[i:i+2],
            data[value_col].iloc[i:i+2],
            color=colormap(i),
            linewidth=1
        )

    # Add title, labels, and grid
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example usage with vertical_daily_aggregates
plot_daily_aggregates(
    data=vertical_daily_aggregates,
    date_col='Date',
    value_col='Average_MEP',
    title="Daily Average MEP Over Time with Different Colors",
    xlabel="Date",
    ylabel="Average_MEP"
)
###############################################################################
###############################################################################
##############Plot the monthly aggregated dataframe############################

# Convert YearMonth to datetime for better handling
aggregated_monthly_data['YearMonth'] = pd.to_datetime(aggregated_monthly_data['YearMonth'])

# Plot the Average_MEP
plt.figure(figsize=(14, 7))
plt.plot(aggregated_monthly_data['YearMonth'], aggregated_monthly_data['Average_MEP'], marker='o', label='Average MEP')

# Format x-axis to show month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month for clarity

plt.title('Average Monthly Electric Price Over Time', fontsize=16)
plt.xlabel('Year-Month', fontsize=14)
plt.ylabel('Average MEP', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(rotation=45)  # Rotate month labels for better readability
plt.show()
###############################################################################
###############################################################################
###########################Vertical Daily Aggregates AC########################
'''
#----------------------------------------------------

####---THIS IS WHERE WE GET THE AC VALUE FOR VERTİCAL DATA"

#----------------------------------------------------
'''

# Ensure 'Date' is datetime and add 'DayOfWeek' column
vertical_daily_aggregates['Date'] = pd.to_datetime(vertical_daily_aggregates['Date'])
vertical_daily_aggregates['DayOfWeek'] = vertical_daily_aggregates['Date'].dt.day_name()

def print_with_dataframe_name_ndiffs(dataframe, dataframe_name, column_name, test_name='adf'):
    """
    Prints the optimal differencing level for the given dataframe and column,
    including the dataframe name, and returns the optimal differencing level.

    Parameters:
        dataframe (pd.DataFrame): The dataframe to analyze.
        dataframe_name (str): The name of the dataframe.
        column_name (str): The column to analyze.
        test_name (str): The test to use for determining differencing (default: 'adf').

    Returns:
        int: The optimal differencing level (d).
    """
    from pmdarima.arima import ndiffs

    d = ndiffs(dataframe[column_name], test=test_name)

    print(f"Optimal differencing level (d) for dataframe '{dataframe_name}', column '{column_name}': {d}")
    return d

def perform_dynamic_differencing(dataframe, metric_col, d):
    """
    Perform dynamic differencing on the specified column of the DataFrame based on the given differencing level.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        metric_col (str): The column name to perform differencing on.
        d (int): The level of differencing to apply.

    Returns:
        tuple: A tuple containing the updated DataFrame and the name of the differenced column.
    """
    differenced_data = dataframe.copy()
    differenced_column_name = f"{metric_col}_diff{d}"
    
    differenced_data[differenced_column_name] = differenced_data[metric_col]
    for i in range(d):
        differenced_data[differenced_column_name] = differenced_data[differenced_column_name].diff()

    differenced_data.dropna(inplace=True)  # Drop NaN values introduced by differencing
    return differenced_data, differenced_column_name

# Function to perform seasonal decomposition
def seasonal_decomposition(data, metric_col, model='additive', period=365):
    """
    Perform seasonal decomposition on the specified column of the DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        metric_col (str): The column name to decompose.
        model (str): The decomposition model, 'additive' or 'multiplicative'.
        period (int): The seasonal period.

    Returns:
        SeasonalDecompose: The decomposition object.
    """
    decomposed = seasonal_decompose(data[metric_col], model=model, period=period)
    
    
    return decomposed

# Function to plot decompositions side by side
def plot_pre_and_post_decomposition(original_data, differenced_data, metric_col, differenced_col, period=24):
    """
    Plot pre-differencing and post-differencing seasonal decomposition side by side.

    Parameters:
        original_data (pd.DataFrame): The original DataFrame.
        differenced_data (pd.DataFrame): The DataFrame after differencing.
        metric_col (str): The column name for the original data.
        differenced_col (str): The column name for the differenced data.
        period (int): The seasonal period for decomposition.
    """
    import matplotlib.pyplot as plt

    # Decompose original data
    decomposed_original = seasonal_decomposition(original_data, metric_col, model='additive', period=period)

    # Decompose differenced data
    decomposed_differenced = seasonal_decomposition(differenced_data, differenced_col, model='additive', period=period)

    # Plot side by side
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Original Data Decomposition
    axes[0, 0].plot(decomposed_original.observed, label='Observed')
    axes[0, 0].set_title('Original: Observed')
    axes[0, 1].plot(decomposed_original.trend, label='Trend')
    axes[0, 1].set_title('Original: Trend')
    axes[0, 2].plot(decomposed_original.seasonal, label='Seasonal')
    axes[0, 2].set_title('Original: Seasonal')
    axes[0, 3].plot(decomposed_original.resid, label='Residual')
    axes[0, 3].set_title('Original: Residual')

    # Differenced Data Decomposition
    axes[1, 0].plot(decomposed_differenced.observed, label='Observed', color='orange')
    axes[1, 0].set_title('Differenced: Observed')
    axes[1, 1].plot(decomposed_differenced.trend, label='Trend', color='orange')
    axes[1, 1].set_title('Differenced: Trend')
    axes[1, 2].plot(decomposed_differenced.seasonal, label='Seasonal', color='orange')
    axes[1, 2].set_title('Differenced: Seasonal')
    axes[1, 3].plot(decomposed_differenced.resid, label='Residual', color='orange')
    axes[1, 3].set_title('Differenced: Residual')

    # Set labels and layout
    for ax in axes.flatten():
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    plt.show()

def analyze_acf_pacf(dataframe, nlags=40, confidence_level=0.99, columns_from_last=3):
    """
    Analyze ACF and PACF for the specified number of columns from the end of a DataFrame, filter significant lags, and plot them.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame with time series data.
        nlags (int): The number of lags to calculate ACF and PACF.
        confidence_level (float): Confidence level for significant lags (e.g., 0.95 or 0.99).
        columns_from_last (int): Number of columns to pick from the end of the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing significant ACF and PACF lags for each column, including additional columns for context.
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf, pacf
    import numpy as np

    # Select the specified number of columns from the end
    selected_columns = dataframe.columns[-columns_from_last:]
    results = []

    # Calculate z-score based on the confidence level
    from scipy.stats import norm
    z_score = norm.ppf((1 + confidence_level) / 2)

    for column in selected_columns:
        data = dataframe[column]
        
        # Calculate ACF and PACF
        lag_acf = acf(data, nlags=nlags, fft=True)
        lag_pacf = pacf(data, nlags=nlags, method='ols')

        # Confidence interval threshold
        n = len(data)
        confidence_threshold = z_score / np.sqrt(n)

        # Filter significant lags
        significant_acf_lags = [(lag, value) for lag, value in enumerate(lag_acf) if abs(value) >= confidence_threshold]
        significant_pacf_lags = [(lag, value) for lag, value in enumerate(lag_pacf) if abs(value) >= confidence_threshold]

        # Append significant lags to results
        for lag, value in significant_acf_lags:
            results.append({
                'Column': column,
                'Type': 'ACF',
                'Lag': lag,
                'Value': value,
                'Confidence Threshold': confidence_threshold
            })

        for lag, value in significant_pacf_lags:
            results.append({
                'Column': column,
                'Type': 'PACF',
                'Lag': lag,
                'Value': value,
                'Confidence Threshold': confidence_threshold
            })

        # Plot ACF and PACF
        plt.figure(figsize=(18, 9))

        # ACF Plot
        plt.subplot(121)
        plt.stem(range(len(lag_acf)), lag_acf)
        plt.axhline(y=confidence_threshold, color='red', linestyle='--', label='Confidence Threshold')
        plt.axhline(y=-confidence_threshold, color='red', linestyle='--')
        plt.title(f'ACF: {column}')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.grid(True)

        # PACF Plot
        plt.subplot(122)
        plt.stem(range(len(lag_pacf)), lag_pacf)
        plt.axhline(y=confidence_threshold, color='red', linestyle='--', label='Confidence Threshold')
        plt.axhline(y=-confidence_threshold, color='red', linestyle='--')
        plt.title(f'PACF: {column}')
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Convert results to a DataFrame
    significant_lags_df = pd.DataFrame(results)

    return significant_lags_df

# Example usage:
# analyze_acf_pacf_with_lag_and_day_names(
#     data=differenced_data_horizontal['Weekday_diff1'],
#     metric_col='Weekday_diff1',
#     start_day='Monday',
#     nlags=60
# )

def augmented_dickey_fuller_test(dataframe, column_name='Average MEP (TL/MWh)_diff2'):
    # Assuming df_detrended is your dataframe with the residuals from the decomposition
    residuals = dataframe[column_name].dropna()  # Replace 'Seasonal_Diff_2' with your residuals column if different

    # Conduct ADF test
    adf_result = adfuller(residuals)

    # Output the results
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    for key, value in adf_result[4].items():
        print(f'Critical Value ({key}): {value}')

    # Determine if data is stationary
    if adf_result[0] < adf_result[4]['5%'] and adf_result[1] < 0.05:
        print("The data is stationary.")
    else:
        print("The data is not stationary.")

if 'Average MEP (TL/MWh)' in vertical_daily_aggregates.columns:
    vertical_daily_aggregates['Average_MEP'] = vertical_daily_aggregates.pop('Average MEP (TL/MWh)')

# Example Usage
original_metric = 'Average MEP'
vertical_daily_aggregates['DateTime'] = pd.to_datetime(vertical_daily_aggregates['Date'])

#Learn how many times we must differentiate our data
d = print_with_dataframe_name_ndiffs(vertical_daily_aggregates, 'vertical_daily_aggregates', 'Average_MEP')

# Perform second-order differencing
differenced_data, differenced_col_name = perform_dynamic_differencing(vertical_daily_aggregates, 'Average_MEP', d)

augmented_dickey_fuller_test(differenced_data, column_name=differenced_col_name)

# Plot pre and post decomposition
plot_pre_and_post_decomposition(
    original_data=vertical_daily_aggregates,
    differenced_data=differenced_data,
    metric_col='Average_MEP',
    differenced_col= 'Average_MEP_diff1',
    period=365
)

# Perform the analysis
significant_lags_vertical = analyze_acf_pacf(
    dataframe=differenced_data,
    nlags=29,
    confidence_level=0.95,
    columns_from_last = 1
)
print(significant_lags_vertical)

###############################################################################
###############################################################################
################Aggregated horizontal hourly ACF/PACF#####################
'''
#----------------------------------------------------

####---THIS IS WHERE WE GET THE AC VALUE FOR HORİZONTAL DATA"

#----------------------------------------------------
'''
# Suppose aggregated_horizontal_data has:
# ['Season', 'DayOfWeek', 'Hour', 'Total_MEP', 'Count', 'Average_MEP', 'TimePeriod', 'Week']
# Ensure that DayOfWeek contains values like 'Monday', 'Tuesday', etc.

# Function to map days to categories
def day_to_group(day):
    if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        return 'Weekday'
    elif day == 'Saturday':
        return 'Saturday'
    else:
        return 'Sunday'

# Step 1: Map each DayOfWeek to a Group
aggregated_horizontal_data['Group'] = aggregated_horizontal_data['DayOfWeek'].apply(day_to_group)

# Step 2: Aggregate by Group, Season, and Hour to compute the mean Average_MEP
grouped = (
    aggregated_horizontal_data
    .groupby(['Group', 'Season', 'Hour'])['Average_MEP']
    .mean()
    .reset_index()
)

# Create a new column 'Season_Hour' by concatenating Season and Hour
grouped['Season_Hour'] = grouped['Season'].astype(str) + '_Hour' + grouped['Hour'].astype(str)

# Step 4: Pivot the DataFrame to have Groups as rows and Season_Hour as columns
result_df = grouped.pivot(index='Group', columns='Season_Hour', values='Average_MEP')

# Step 5: Order the columns by Season and then Hour
seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
hours_order = list(range(24))  # 0 to 23
ordered_columns = [f'{season}_Hour{hour}' for season in seasons_order for hour in hours_order]

result_df = result_df.reindex(columns=ordered_columns)

print("Final Aggregated DataFrame:")
print(result_df)

# Transpose the dataframe
transposed_horizontal_df = result_df.transpose()

# Extract season and hour information from the index (column names of original dataframe)
transposed_horizontal_df.reset_index(inplace=True)

# Split the 'Season_Hour' column into 'Season' and 'Hour'
transposed_horizontal_df[['Season', 'Hour']] = transposed_horizontal_df['Season_Hour'].str.extract(r'(?P<Season>\w+)_Hour(?P<Hour>\d+)')

# Convert Hour to integer for analysis purposes
transposed_horizontal_df['Hour'] = transposed_horizontal_df['Hour'].astype(int)

# Reorder columns to have 'Season', 'Hour', and then group values
columns = ['Season', 'Hour'] + [col for col in transposed_horizontal_df.columns if col not in ['index', 'Season', 'Hour']]
transposed_horizontal_df = transposed_horizontal_df[columns]

# Drop the 'Season_Hour' column
transposed_horizontal_df.drop(columns=['Season_Hour'], inplace=True)

# Reorder columns to place 'Weekday' before 'Saturday'
columns_order = ['Season', 'Hour', 'Weekday', 'Saturday', 'Sunday'] + [col for col in transposed_horizontal_df.columns if col not in ['Season', 'Hour', 'Weekday', 'Saturday', 'Sunday']]
transposed_horizontal_df = transposed_horizontal_df[columns_order]

# Find the best Number for differencing
# Perform differencing
# Conduct ADF Test

d = print_with_dataframe_name_ndiffs(transposed_horizontal_df, 'transposed_horizontal_df', 'Weekday', test_name='adf')
differenced_data_horizontal, differenced_col_name = perform_dynamic_differencing(transposed_horizontal_df, 'Saturday', d)
augmented_dickey_fuller_test(differenced_data_horizontal, column_name=differenced_col_name)

d = print_with_dataframe_name_ndiffs(transposed_horizontal_df, 'transposed_horizontal_df', 'Saturday', test_name='adf')
differenced_data_horizontal, differenced_col_name = perform_dynamic_differencing(differenced_data_horizontal, 'Sunday', d)
augmented_dickey_fuller_test(differenced_data_horizontal, column_name=differenced_col_name)

d = print_with_dataframe_name_ndiffs(transposed_horizontal_df, 'transposed_horizontal_df', 'Sunday', test_name='adf')
differenced_data_horizontal, differenced_col_name = perform_dynamic_differencing(differenced_data_horizontal, 'Weekday', d)
augmented_dickey_fuller_test(differenced_data_horizontal, column_name=differenced_col_name)

last_column_name_weekday = differenced_data_horizontal.columns[-1]
last_column_name_saturday = differenced_data_horizontal.columns[-3]
last_column_name_sunday = differenced_data_horizontal.columns[-2]


# Plot pre and post decomposition
plot_pre_and_post_decomposition(
    original_data=transposed_horizontal_df,
    differenced_data=differenced_data_horizontal,
    metric_col='Weekday',
    differenced_col= last_column_name_weekday,
    period=24
)

# Plot pre and post decomposition
plot_pre_and_post_decomposition(
    original_data=transposed_horizontal_df,
    differenced_data=differenced_data_horizontal,
    metric_col='Saturday',
    differenced_col= last_column_name_saturday,
    period=24
)

# Plot pre and post decomposition
plot_pre_and_post_decomposition(
    original_data=transposed_horizontal_df,
    differenced_data=differenced_data_horizontal,
    metric_col='Sunday',
    differenced_col= last_column_name_sunday,
    period=24
)


# Call the updated ACF/PACF analysis function
# Parameters for analysis
nlags = 23  # Number of lags to analyze
confidence_level = 0.95  # Confidence level for filtering significant lags

# Perform the analysis
significant_lags_horizontal = analyze_acf_pacf(
    dataframe=differenced_data_horizontal,
    nlags=nlags,
    confidence_level=confidence_level,
    columns_from_last = 3
)

# Output the results
print(significant_lags_horizontal)
##########################################################################################
##########################################################################################
##########################################################################################
'''
#----------------------------------------------------

####---Aggregated Monthly Data AC Value

#----------------------------------------------------
'''

#Learn how many times we must differentiate our data
d = print_with_dataframe_name_ndiffs(aggregated_monthly_data, 'aggregated_monthly_data', 'Average_MEP')
print(f"Optimal differencing level (d): {d}")

# Perform second-order differencing
differenced_data_monthly, differenced_col_name = perform_dynamic_differencing(aggregated_monthly_data, 'Average_MEP', d)
# Get the last column's name
last_column_name = differenced_data_monthly.columns[-1]
augmented_dickey_fuller_test(differenced_data_monthly, column_name=last_column_name)


# Plot pre and post decomposition
plot_pre_and_post_decomposition(
    original_data=aggregated_monthly_data,
    differenced_data=differenced_data_monthly,
    metric_col='Average_MEP',
    differenced_col= last_column_name,
    period = 12
)

# Call the updated ACF/PACF analysis function
# Parameters for analysis
nlags = 11  # Number of lags to analyze
confidence_level = 0.95  # Confidence level for filtering significant lags


# Perform the analysis
significant_lags_monthly = analyze_acf_pacf(
    dataframe=differenced_data_monthly,
    nlags=nlags,
    confidence_level=confidence_level,
    columns_from_last = 1
)
########################################################################################
########################################################################################
'''
#-----------------------------------------------------------------------------#
#            Aggregated vertical daily data seasonality analysis              # 
#-----------------------------------------------------------------------------#
'''
# Step 1: Ensure 'Date' is in datetime format
vertical_daily_aggregates['Date'] = pd.to_datetime(vertical_daily_aggregates['Date'])

# Assuming vertical_daily_aggregates is your DataFrame
vertical_daily_aggregates.rename(columns={'Average MEP (TL/MWh)': 'Average_MEP'}, inplace=True)

# Step 2: Apply STL decomposition
stl = STL(vertical_daily_aggregates['Average_MEP'], period=7)  # Weekly seasonality
result = stl.fit()
seasonal = result.seasonal

# Step 3: Add DayOfWeek and Month information to the DataFrame
vertical_daily_aggregates['Seasonal Component'] = seasonal
vertical_daily_aggregates['DayOfWeek'] = vertical_daily_aggregates['Date'].dt.day_name()
vertical_daily_aggregates['Month'] = vertical_daily_aggregates['Date'].dt.month

# Step 4: Aggregate weekly seasonality by Month
monthly_weekly_seasonality = {}
for month in range(1, 13):  # Months 1 to 12
    # Filter data for the current month
    monthly_data = vertical_daily_aggregates[vertical_daily_aggregates['Month'] == month]
    
    # Group by day of the week and calculate the mean seasonal component
    weekly_seasonality = monthly_data.groupby('DayOfWeek')['Seasonal Component'].mean()
    
    # Ensure the days are in the correct order (Monday to Sunday)
    weekly_seasonality = weekly_seasonality.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                                     'Friday', 'Saturday', 'Sunday'])
    
    # Store in dictionary using calendar's month name
    monthly_weekly_seasonality[calendar.month_name[month]] = weekly_seasonality

# Step 5: Compute overall average weekly seasonality
overall_weekly_seasonality = pd.DataFrame(monthly_weekly_seasonality).mean(axis=1)

# Step 6: Plot aggregated weekly seasonality for each month + overall
n_plots = len(monthly_weekly_seasonality) + 1  # Total number of plots: 12 months + 1 overall
fig, axes = plt.subplots((n_plots // 3) + (n_plots % 3 > 0), 3, figsize=(18, 15))  # Dynamic grid layout
axes = axes.flatten()

# Plot each month's weekly seasonality
for idx, (month, weekly_seasonality) in enumerate(monthly_weekly_seasonality.items()):
    ax = axes[idx]
    ax.plot(weekly_seasonality.index, weekly_seasonality.values, marker='o', linestyle='-', color='green')
    ax.set_title(f"{month}")
    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Avg Seasonal Component")
    ax.grid()

# Plot the overall weekly seasonality
ax = axes[len(monthly_weekly_seasonality)]
ax.plot(overall_weekly_seasonality.index, overall_weekly_seasonality.values, marker='o', 
        linestyle='-', color='blue')
ax.set_title("Overall Weekly Seasonality")
ax.set_xlabel("Day of the Week")
ax.set_ylabel("Avg Seasonal Component")
ax.grid()

# Remove unused subplots
for idx in range(len(monthly_weekly_seasonality) + 1, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.suptitle("Aggregated Weekly Seasonality for Each Month + Overall", fontsize=18, y=1.02)
plt.show()

# Step 7: Print aggregated results
print("Aggregated Weekly Seasonality (Per Month):")
for month, weekly_seasonality in monthly_weekly_seasonality.items():
    print(f"\n{month}:\n{weekly_seasonality}")

overall_weekly_seasonality = overall_weekly_seasonality.to_frame(name="Seasonality Component")
overall_weekly_seasonality.reset_index(inplace=True)
overall_weekly_seasonality.columns = ["Weekday", "Seasonality Component"]

print("\nOverall Weekly Seasonality:")
print(overall_weekly_seasonality)
###############################################################################
###############################################################################
################Aggregated horizontal hourly data seasonality analysis################ 
###############################################################################
'''
#------------------------------------------------------------------------------------------------#
#-----------------------STL appraoch to aggregated_horizontal data_seasonaltiy-------------------#
#------------------------------------------------------------------------------------------------#
'''

# Initialize seasonality DataFrame
seasonality_by_day_period = pd.DataFrame()

# Analyze STL seasonality for the provided result_df
def analyze_stl_seasonality(result_df):
    global seasonality_by_day_period

    try:
        # Create a DataFrame to store seasonality data
        seasonality_data = []

        # Loop over each row (Group: Saturday, Sunday, Weekday)
        for group, row in result_df.iterrows():
            # Extract the 96-hour data for the group
            group_data = row.values

            # Ensure sufficient data for decomposition
            if len(group_data) >= 96:
                stl = STL(group_data, period=24)
                result = stl.fit()

                # Calculate the average seasonal component for each hour (0-23)
                seasonal_avg = [np.mean(result.seasonal[hour::24]) for hour in range(24)]

                # Append the averaged seasonal component for 24 hours to the list
                for hour, seasonal in enumerate(seasonal_avg):
                    seasonality_data.append({
                        'Group': group,
                        'Hour': hour,
                        'Seasonal_Component': seasonal
                    })

                # Plot the seasonal component for the averaged 24 hours
                plt.figure(figsize=(10, 6))
                plt.plot(seasonal_avg, label=f"Seasonality - {group}", color=sns.color_palette("hsv", 3)[['Saturday', 'Sunday', 'Weekday'].index(group)])
                plt.title(f"Seasonality for {group} (Averaged 24 Hours)")
                plt.xlabel("Hour")
                plt.ylabel("Seasonal Component")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()

        # Convert the seasonality data list to a DataFrame
        seasonality_by_day_period = pd.DataFrame(seasonality_data)

        # Display the seasonality DataFrame
        print("Stored Seasonality by Day and Period:")
        print(seasonality_by_day_period.head())

    except Exception as e:
        print(f"Error during STL seasonality analysis: {e}")

# Example usage for STL seasonality analysis
analyze_stl_seasonality(result_df)
seasonality_by_day_period 

'''
#------------------------------------------------------------------------------------------------#
#---------------Trigonometric appraoch toe aggregated_horizontal data_seasonaltiy----------------#
#------------------------------------------------------------------------------------------------#
'''
# Prepare dictionary to store seasonality results for horizontally aggregated data
horizontal_seasonality_results = {}

def calculate_daynight_annual_seasonality(data):
    """
    Analyze day-night and annual seasonality using trigonometric functions.
    """
    global horizontal_seasonality_results

    unique_seasons = data['Season'].unique()
    unique_days = data['DayOfWeek'].unique()

    for season in unique_seasons:
        for day in unique_days:
            # Filter data for the specific season and day
            subset = data[(data['Season'] == season) & (data['DayOfWeek'] == day)]

            if subset.empty:
                continue

            # Day cycle: Normalize Hour for day (0:00 to 12:00)
            t_day = subset[subset['Hour'] < 12]['Hour'] / 12
            y_day = subset[subset['Hour'] < 12]['Average_MEP']

            # Night cycle: Normalize Hour for night (12:00 to 24:00)
            t_night = (subset[subset['Hour'] >= 12]['Hour'] - 12) / 12
            y_night = subset[subset['Hour'] >= 12]['Average_MEP']

            # Annual cycle: Use normalized hours and seasonal grouping
            t_year = subset['Hour'] / 24
            y_year = subset['Average_MEP']

            # Design matrices for day, night, and annual cycles
            X_day = np.column_stack([
                np.sin(2 * np.pi * t_day),
                np.cos(2 * np.pi * t_day),
                np.ones_like(t_day)  # Intercept term
            ])
            X_night = np.column_stack([
                np.sin(2 * np.pi * t_night),
                np.cos(2 * np.pi * t_night),
                np.ones_like(t_night)  # Intercept term
            ])
            X_year = np.column_stack([
                np.sin(2 * np.pi * t_year),
                np.cos(2 * np.pi * t_year),
                np.ones_like(t_year)  # Intercept term
            ])

            # Fit regression models for day, night, and annual cycles
            day_model = LinearRegression().fit(X_day, y_day)
            night_model = LinearRegression().fit(X_night, y_night)
            year_model = LinearRegression().fit(X_year, y_year)

            # Store the results
            horizontal_seasonality_results[f"{season}-{day}"] = {
                "Day": {
                    "s1 (sin 2πt_day)": day_model.coef_[0],
                    "s2 (cos 2πt_day)": day_model.coef_[1],
                    "Intercept (s3_day)": day_model.intercept_
                },
                "Night": {
                    "s1 (sin 2πt_night)": night_model.coef_[0],
                    "s2 (cos 2πt_night)": night_model.coef_[1],
                    "Intercept (s3_night)": night_model.intercept_
                },
                "Annual": {
                    "s1 (sin 2πt_year)": year_model.coef_[0],
                    "s2 (cos 2πt_year)": year_model.coef_[1],
                    "Intercept (s3_year)": year_model.intercept_
                }
            }

            # Debugging: Print coefficients for each combination
            print(f"Seasonality coefficients for {season} - {day}:")
            print(horizontal_seasonality_results[f"{season}-{day}"])

    return horizontal_seasonality_results

# Analyze seasonality
horizontal_seasonality_results = calculate_daynight_annual_seasonality(aggregated_horizontal_hourly_data)

# Convert the `horizontal_seasonality_results` dictionary to a pandas DataFrame
trigonometric_horizontal_seasonality_data = []

for key, values in horizontal_seasonality_results.items():
    season, day = key.split('-')
    day_data = {
        'Season': season,
        'Day': day,
        'Day_s1 (sin 2πt_day)': values['Day']['s1 (sin 2πt_day)'],
        'Day_s2 (cos 2πt_day)': values['Day']['s2 (cos 2πt_day)'],
        'Day_Intercept': values['Day']['Intercept (s3_day)'],
        'Night_s1 (sin 2πt_night)': values['Night']['s1 (sin 2πt_night)'],
        'Night_s2 (cos 2πt_night)': values['Night']['s2 (cos 2πt_night)'],
        'Night_Intercept': values['Night']['Intercept (s3_night)'],
        'Annual_s1 (sin 2πt_year)': values['Annual']['s1 (sin 2πt_year)'],
        'Annual_s2 (cos 2πt_year)': values['Annual']['s2 (cos 2πt_year)'],
        'Annual_Intercept': values['Annual']['Intercept (s3_year)']
    }
    trigonometric_horizontal_seasonality_data.append(day_data)

# Create the DataFrame
trigonometric_horizontal_seasonality_data = pd.DataFrame(trigonometric_horizontal_seasonality_data)

# Ensure only numeric columns are used for aggregation
numeric_columns = trigonometric_horizontal_seasonality_data.select_dtypes(include=['number']).columns

# Add 'Day' column to the numeric columns for grouping purposes
columns_to_group = ['Day'] + list(numeric_columns)
filtered_data = trigonometric_horizontal_seasonality_data[columns_to_group]

# Group by 'Day' and compute the mean for numeric columns
trigonometric_horizontal_seasonality_data = filtered_data.groupby('Day', as_index=False).mean()

# Ensure 'Day' column is treated as a categorical type with the specified order
trigonometric_horizontal_seasonality_data['Day'] = pd.Categorical(trigonometric_horizontal_seasonality_data['Day'], categories=weekdays, ordered=True)

# Sort the DataFrame by the 'Day' column
trigonometric_horizontal_seasonality_data = trigonometric_horizontal_seasonality_data.sort_values(by='Day').reset_index(drop=True)

# Display the aggregated DataFrame
print(trigonometric_horizontal_seasonality_data)
# Visualize seasonality trends for each day (aggregated data)
for index, row in trigonometric_horizontal_seasonality_data.iterrows():
    day = row['Day']

    # Simulate hour data for day and night periods
    t_day = np.linspace(0, 1, 12)  # Normalized hours for day (0 to 12)
    t_night = np.linspace(0, 1, 12)  # Normalized hours for night (12 to 24)
    t_year = np.linspace(0, 1, 24)  # Normalized hours for annual cycle

    # Calculate seasonality components
    try:
        seasonality_day = (
            row['Day_s1 (sin 2πt_day)'] * np.sin(2 * np.pi * t_day) +
            row['Day_s2 (cos 2πt_day)'] * np.cos(2 * np.pi * t_day) +
            row['Day_Intercept']
        )

        seasonality_night = (
            row['Night_s1 (sin 2πt_night)'] * np.sin(2 * np.pi * t_night) +
            row['Night_s2 (cos 2πt_night)'] * np.cos(2 * np.pi * t_night) +
            row['Night_Intercept']
        )

        seasonality_year = (
            row['Annual_s1 (sin 2πt_year)'] * np.sin(2 * np.pi * t_year) +
            row['Annual_s2 (cos 2πt_year)'] * np.cos(2 * np.pi * t_year) +
            row['Annual_Intercept']
        )
    except KeyError as e:
        print(f"Missing key in data for {day}: {e}")
        continue

    # Visualization
    plt.figure(figsize=(14, 8))

    # Plot day seasonality
    plt.plot(t_day * 12, seasonality_day, label="Day Seasonality Fit", color='blue', linewidth=2)

    # Plot night seasonality
    plt.plot((t_night * 12) + 12, seasonality_night, label="Night Seasonality Fit", color='orange', linewidth=2)

    # Plot annual seasonality
    plt.plot(t_year * 24, seasonality_year, label="Annual Seasonality Fit", color='green', linewidth=2)

    plt.axhline(y=np.mean([row['Day_Intercept'], row['Night_Intercept'], row['Annual_Intercept']]),
                color='red', linestyle='--', label='Mean Seasonal Value')
    plt.title(f"Seasonality for {day}", fontsize=16)
    plt.xlabel("Time (Hours)", fontsize=12)
    plt.ylabel("Seasonal Component (TL/MWh)", fontsize=12)
    plt.xticks(ticks=np.arange(0, 25, 3), labels=[f'{i}:00' for i in range(0, 25, 3)], fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.show()

###############################################################################
###############################################################################
'''
#-----------------------------------------------------------------------------#
#                Aggregated monthly data seasonality analysis                 # 
#-----------------------------------------------------------------------------#
'''
# DataFrame to store monthly seasonality
monthly_seasonality = pd.DataFrame(columns=['Month', 'Seasonal_Component'])

# STL Decomposition to analyze seasonality for 12 unique months
def analyze_monthly_seasonality_with_stl(data, value_col='Average_MEP', period=12):
    global monthly_seasonality

    try:
        # Use index if YearMonth is not a column
        if 'YearMonth' not in data.columns and data.index.name == 'YearMonth':
            data = data.reset_index()

        # Ensure the YearMonth column is datetime
        if 'YearMonth' in data.columns:
            data['YearMonth'] = pd.to_datetime(data['YearMonth'])
        else:
            raise KeyError("The column 'YearMonth' is not in the DataFrame or index.")

        data = data.set_index('YearMonth')

        # STL decomposition
        stl = STL(data[value_col], period=period)
        stl_result = stl.fit()

        # Extract seasonal components for each month
        data['Seasonal'] = stl_result.seasonal
        seasonality_list = []
        month_names = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        for month in range(1, 13):
            month_data = data[data.index.month == month]
            if not month_data.empty:
                seasonality_list.append({'Month': month_names[month - 1], 'Seasonal_Component': month_data['Seasonal'].mean()})

        # Update the global DataFrame
        monthly_seasonality = pd.DataFrame(seasonality_list).sort_values(by='Month', key=lambda x: pd.Categorical(x, categories=month_names, ordered=True)).reset_index(drop=True)

        # Plot the seasonal component for 12 unique months
        colors = sns.color_palette("husl", 12)  # Generate 12 distinct colors
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(1, 13),
            monthly_seasonality['Seasonal_Component'],
            color=colors,
            tick_label=month_names
        )
        plt.title('Seasonality by Month (STL Decomposition)')
        plt.xlabel('Month')
        plt.ylabel('Average Seasonal Component')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        # Print the DataFrame
        print("Seasonality Component for Each Month:")
        print(monthly_seasonality)

        return monthly_seasonality

    except Exception as e:
        print(f"Error during STL decomposition: {e}")

# Example usage for STL seasonality analysis
monthly_seasonality = analyze_monthly_seasonality_with_stl(
    data=aggregated_monthly_data,
    value_col='Average_MEP',
    period=12  # Monthly seasonality
)
###############################################################################
###############################################################################
'''
#-----------------------------------------------------------------------------#
#                  Finding the Necessary Distributions                        # 
#-----------------------------------------------------------------------------#
'''
# --- Initialize Empty DataFrames for Results ---
hourly_results_df = pd.DataFrame(columns=['Hour', 'Best_Distribution', 'KS_Statistic', 'Parameters', 'Parameter_Explanation'])
weekday_results_df = pd.DataFrame(columns=['Weekday', 'Best_Distribution', 'KS_Statistic', 'Parameters', 'Parameter_Explanation'])
monthly_results_df = pd.DataFrame(columns=['Month', 'Best_Distribution', 'KS_Statistic', 'Parameters', 'Parameter_Explanation'])
# --- Group Data for Each Hour ---
hourly_data_df = (
    filtered_analysis_dataframe.groupby('Hour')['MEP (TL/MWh)']
    .apply(list)
    .reset_index(name='Hourly_MEP')
)

# --- Group Data for Each Weekday ---
weekday_data_df = (
    filtered_analysis_dataframe.groupby('DayOfWeek')['MEP (TL/MWh)']
    .apply(list)
    .reset_index(name='Weekday_MEP')
)

# Ensure MonthName is derived correctly
aggregated_monthly_data['MonthName'] = pd.to_datetime(aggregated_monthly_data['YearMonth']).dt.month_name()

# --- Group Data for Each Month ---
monthly_data_df = (
    aggregated_monthly_data.groupby('MonthName')['Average_MEP']
    .apply(list)
    .reset_index(name='Monthly_MEP')
    .sort_values(by='MonthName', key=lambda col: col.map(lambda x: list(calendar.month_name).index(x)))
)


def ks_test(data, distributions):
    """
    Performs the Kolmogorov-Smirnov test for a given dataset against multiple distributions
    and returns the best-fitting distribution based on the lowest KS statistic.
    """
    results = []
    data = pd.Series(data).dropna()  # Convert to pandas Series to use dropna
    data = np.clip(data, -1e10, 1e10)  # Clip to safe range
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    for dist in distributions:
        try:
            params = dist.fit(data)
            cdf_fitted = dist.cdf(sorted_data, *params)
            statistic = np.max(np.abs(ecdf - cdf_fitted))
            param_explanation = get_parameter_explanation(dist.name, params)
            results.append({
                'Distribution': dist.name,
                'KS_Statistic': statistic,
                'Parameters': params,
                'Parameter_Explanation': param_explanation
            })
        except Exception:
            results.append({
                'Distribution': dist.name,
                'KS_Statistic': np.inf,
                'Parameters': None,
                'Parameter_Explanation': "Fit failed"
            })
    return sorted(results, key=lambda x: x['KS_Statistic'])[0]


# --- Helper Function to Explain Parameters ---
def get_parameter_explanation(distribution_name, params):
    explanations = {
        'norm': "(mean, standard deviation)",
        'lognorm': "(shape, location, scale)",
        'expon': "(location, scale)",
        'weibull_min': "(shape, location, scale)",
        'gamma': "(shape, location, scale)",
        'beta': "(shape1, shape2, location, scale)",
        'chi2': "(degrees of freedom, location, scale)",
        'cauchy': "(location, scale)",
        'laplace': "(location, scale)",
        'uniform': "(location, scale)",
        'pareto': "(shape, location, scale)",
        'logistic': "(location, scale)",
        'rayleigh': "(scale)",
        'gumbel_l': "(location, scale)",
        'gumbel_r': "(location, scale)",
        't': "(degrees of freedom, location, scale)",
        'f': "(dfn, dfd, scale)",
        'powerlaw': "(shape, location, scale)",
        'triang': "(shape, location, scale)",
        'genextreme': "(shape, location, scale)",
        'genpareto': "(shape, location, scale)",
        'levy': "(location, scale)",
        'fisk': "(shape, location, scale)",
        'nakagami': "(shape, location, scale)",
        'vonmises': "(kappa, location)",
        'truncnorm': "(lower bound, upper bound, location, scale)"
    }
    return explanations.get(distribution_name, "Parameters not documented")

# Perform KS Test for Hourly Data
hourly_results_df = pd.DataFrame(columns=['Hour', 'Best_Distribution', 'KS_Statistic', 'Parameters', 'Parameter_Explanation'])
for _, row in hourly_data_df.iterrows():
    hour = row['Hour']
    data = row['Hourly_MEP']
    best_fit = ks_test(data, distributions)
    hourly_results_df = pd.concat([hourly_results_df, pd.DataFrame([{
        'Hour': hour,
        'Best_Distribution': best_fit['Distribution'],
        'KS_Statistic': best_fit['KS_Statistic'],
        'Parameters': best_fit['Parameters'],
        'Parameter_Explanation': best_fit['Parameter_Explanation']
    }])], ignore_index=True)

# Perform KS Test for Weekday Data
weekday_results_df = pd.DataFrame(columns=['Weekday', 'Best_Distribution', 'KS_Statistic', 'Parameters', 'Parameter_Explanation'])
for _, row in weekday_data_df.iterrows():
    weekday = row['DayOfWeek']
    data = row['Weekday_MEP']
    best_fit = ks_test(data, distributions)
    weekday_results_df = pd.concat([weekday_results_df, pd.DataFrame([{
        'Weekday': weekday,
        'Best_Distribution': best_fit['Distribution'],
        'KS_Statistic': best_fit['KS_Statistic'],
        'Parameters': best_fit['Parameters'],
        'Parameter_Explanation': best_fit['Parameter_Explanation']
    }])], ignore_index=True)

# Perform KS Test for Monthly Data
monthly_results_df = pd.DataFrame(columns=['Month', 'Best_Distribution', 'KS_Statistic', 'Parameters', 'Parameter_Explanation'])
for _, row in monthly_data_df.iterrows():
    month = row['MonthName']
    data = row['Monthly_MEP']
    best_fit = ks_test(data, distributions)
    monthly_results_df = pd.concat([monthly_results_df, pd.DataFrame([{
        'Month': month,
        'Best_Distribution': best_fit['Distribution'],
        'KS_Statistic': best_fit['KS_Statistic'],
        'Parameters': best_fit['Parameters'],
        'Parameter_Explanation': best_fit['Parameter_Explanation']
    }])], ignore_index=True)

# --- Sort DataFrames for Better Readability ---
# Sort hourly results
hourly_results_df = hourly_results_df.sort_values(by='Hour').reset_index(drop=True)

# Sort weekday results
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_results_df['Weekday'] = pd.Categorical(weekday_results_df['Weekday'], categories=weekday_order, ordered=True)
weekday_results_df = weekday_results_df.sort_values(by='Weekday').reset_index(drop=True)

# Sort monthly results
month_order = list(calendar.month_name)[1:]  # ['January', ..., 'December']
monthly_results_df['Month'] = pd.Categorical(monthly_results_df['Month'], categories=month_order, ordered=True)
monthly_results_df = monthly_results_df.sort_values(by='Month').reset_index(drop=True)

# --- Display Results ---
print("Hourly Best-Fit Results:")
print(hourly_results_df.to_string(index=False))

print("\nWeekday Best-Fit Results:")
print(weekday_results_df.to_string(index=False))

print("\nMonthly Best-Fit Results:")
print(monthly_results_df.to_string(index=False))


#####################################################################################
#####################################################################################
#-----------------------------------------------------------------------------#
#                          Monte Carlo Simulation                             # 
#-----------------------------------------------------------------------------#
def apply_variation_with_console(results_df):
    """
    Adjusts scale and mean parameters of a DataFrame interactively through console input.

    Parameters:
        results_df (pd.DataFrame): Original DataFrame with distribution parameters.

    Returns:
        pd.DataFrame: Adjusted DataFrame with updated scale and mean parameters.
    """
    # Make a deep copy of the DataFrame to avoid altering the original data
    adjusted_results_df = copy.deepcopy(results_df)

    # Ask user if they want to scale the variance
    scale_variance = input("Do you want to scale the variance? (Yes/No): ").strip().lower()
    scaling_factor = 1.0

    if scale_variance == "yes":
        scaling_percentage = float(input("Enter scaling percentage (e.g., 0.1 for +10%): ").strip())
        scaling_factor += scaling_percentage  # Convert percentage to scaling multiplier

    # Ask user if they want to adjust the mean
    adjust_mean = input("Do you want to adjust the mean? (Yes/No): ").strip().lower()
    mean_adjustment = 0.0

    if adjust_mean == "yes":
        mean_adjustment = float(input("Enter mean adjustment value (e.g., 0.2 to add 0.2 to the mean): ").strip())

    # Loop through each row to adjust the scale and mean parameters
    for index, row in adjusted_results_df.iterrows():
        dist_name = row['Best_Distribution']
        params = row['Parameters']

        if isinstance(params, str):
            # Convert string representation of parameters to a tuple
            params = eval(params)

        # Adjust scale and mean (location) parameters
        if dist_name in ['genextreme', 'lognorm']:  # Distributions with (shape, loc, scale)
            adjusted_params = (params[0], params[1] + mean_adjustment, params[2] * scaling_factor)
        elif dist_name in ['gumbel_r']:  # Distributions with (loc, scale)
            adjusted_params = (params[0] + mean_adjustment, params[1] * scaling_factor)
        else:
            # Default case assumes location is the second-to-last parameter
            adjusted_params = params[:-2] + (params[-2] + mean_adjustment, params[-1] * scaling_factor)

        # Update the parameters in the DataFrame
        adjusted_results_df.at[index, 'Parameters'] = adjusted_params

    return adjusted_results_df

# Example Usage:
#scaled_hourly_results_df = apply_variation_with_console(hourly_results_df)

'''
#-----------------------------------------------------------------------------#
#                     Simulating Average Monthly Prices                       # 
#-----------------------------------------------------------------------------#
'''
def simulate_monthly_prices_with_pacf(
    monthly_seasonality, significant_lags_monthly, monthly_results_df, num_months
):
    """
    Simulates monthly electricity prices using Monte Carlo simulation with PACF values.

    Parameters:
        monthly_seasonality (pd.DataFrame): DataFrame with monthly seasonal components.
        significant_lags_monthly (pd.DataFrame): DataFrame with PACF values.
        monthly_results_df (pd.DataFrame): DataFrame with distribution fitting results.
        num_months (int): Number of months to simulate.

    Returns:
        pd.DataFrame: Simulated monthly electricity prices.
    """
    import ast  # For safer parsing of parameters
    import numpy as np

    # Extract seasonal components
    seasonal_component = monthly_seasonality['Seasonal_Component'].values

    # Initialize autocorrelation coefficients dynamically
    autocorrelation_coeff = np.zeros(num_months)
    for i, month in enumerate(monthly_seasonality['Month']):
        dynamic_key = None
        for diff_type in ['diff1', 'diff2', 'diff3']:  # Extend with additional diff types if needed
            key = f"{month}_{diff_type}"
            if key in significant_lags_monthly['Column'].values:
                dynamic_key = key
                break
        if dynamic_key:
            autocorrelation_coeff[i] = significant_lags_monthly.loc[
                significant_lags_monthly['Column'] == dynamic_key, 'Value'
            ].values[0]

    # Extract error distributions from monthly_results_df
    empirical_errors = []
    for _, row in monthly_results_df.iterrows():
        dist_name = row['Best_Distribution']
        params = row['Parameters']
        if isinstance(params, str):
            params = ast.literal_eval(params)  # Safely parse string to tuple
        dist = globals()[dist_name]  # Dynamically get the distribution object
        samples = dist.rvs(*params, size=1000)  # Generate multiple samples from the distribution
        empirical_errors.append(samples)

    # Initialize simulation
    prices = np.zeros(num_months)
    for m in range(num_months):
        if m == 0:
            # Initialize with the first month (assume mean seasonal component + error)
            while True:
                simulated_price = seasonal_component[m] + np.random.choice(empirical_errors[m])
                if simulated_price > 0:
                    prices[m] = simulated_price
                    break
        else:
            # Apply the simulation equation, retry until the simulated price is greater than 0
            while True:
                pacf_contribution = 0
                for _, pacf_row in significant_lags_monthly[
                    significant_lags_monthly['Column'] == f"{monthly_seasonality['Month'][m]}_diff1"
                ].iterrows():
                    lag = pacf_row['Lag']
                    pacf_value = pacf_row['Value']
                    adjusted_month = m - lag
                    if adjusted_month >= 0:
                        pacf_contribution += pacf_value * (
                            prices[adjusted_month] - seasonal_component[adjusted_month]
                        )

                simulated_price = (
                    seasonal_component[m]
                    + pacf_contribution
                    + np.random.choice(empirical_errors[m])
                )
                if simulated_price > 0:
                    prices[m] = simulated_price
                    break

    # Create DataFrame for results
    simulated_monthly_df = pd.DataFrame({
        'Month': monthly_seasonality['Month'],
        'Simulated_Price': prices
    })

    return simulated_monthly_df
# Example usage
num_months = 12
simulated_monthly_prices = simulate_monthly_prices_with_pacf(
    monthly_seasonality=monthly_seasonality,
    significant_lags_monthly=significant_lags_monthly,
    monthly_results_df=monthly_results_df,
    num_months=num_months
)
print(simulated_monthly_prices)

'''
#-----------------------------------------------------------------------------#
#                     Simulating Average Daily Prices                         # 
#-----------------------------------------------------------------------------#
'''
def simulate_daily_prices_with_pacf_sampling(
    monthly_simulated_prices, overall_weekly_seasonality, significant_lags_vertical, weekday_results_df
):
    """
    Simulates daily average electricity prices using Monte Carlo simulation with PACF values.
    """
    import ast  # For safer parsing of parameters
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # Debug: Ensure weekday_results_df contains the correct data
    #print("weekday_results_df columns:", weekday_results_df.columns)
    #print("weekday_results_df sample:", weekday_results_df.head())

    # Extract seasonality for each weekday
    weekday_seasonality = overall_weekly_seasonality.set_index('Weekday')['Seasonality Component'].to_dict()

    # Extract PACF values dynamically for daily data
    pacf_lags = significant_lags_vertical

    # Extract error distributions for each weekday
    empirical_errors = {}
    for _, row in weekday_results_df.iterrows():
        try:
            dist_name = row['Best_Distribution']
            params = row['Parameters']
            if isinstance(params, str):
                params = ast.literal_eval(params)
            dist = globals()[dist_name]
            empirical_errors[row['Weekday']] = (dist, params)
        except KeyError as e:
            print(f"KeyError during empirical_errors generation: {e}")
            raise

    # Generate dates for the year
    start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-12-31", "%Y-%m-%d")
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # Debug: Log generated dates
    #print("Generated dates:", all_dates[:5])

    # Initialize simulation
    simulated_daily_prices = []
    daily_prices_by_month = {month: [] for month in monthly_simulated_prices['Month']}

    for date in all_dates:
        month = date.strftime("%B")
        weekday = date.strftime("%A")
        avg_monthly_price = monthly_simulated_prices.loc[
            monthly_simulated_prices['Month'] == month, 'Simulated_Price'
        ].values[0]

        # Debug: Check weekday and month alignment
        #print(f"Processing date: {date}, weekday: {weekday}, month: {month}")

        max_retries = 1000
        retry_count = 0
        while retry_count < max_retries:
            try:
                pacf_contribution = 0
                for _, pacf_row in pacf_lags[pacf_lags['Column'] == f"{weekday}_diff1"].iterrows():
                    lag = pacf_row['Lag']
                    pacf_value = pacf_row['Value']
                    adjusted_day = len(simulated_daily_prices) - lag
                    if adjusted_day >= 0:
                        pacf_contribution += pacf_value * (
                            simulated_daily_prices[adjusted_day][2] - weekday_seasonality[weekday]
                        )
                    elif len(simulated_daily_prices) > abs(adjusted_day):
                        pacf_contribution += pacf_value * (
                            simulated_daily_prices[-lag][2] - weekday_seasonality[weekday]
                        )

                seasonal_component = weekday_seasonality[weekday]
                dist, params = empirical_errors[weekday]

                simulated_price = (
                    seasonal_component + pacf_contribution + dist.rvs(*params, size=1)[0]
                )

                if simulated_price > 0:
                    daily_prices_by_month[month].append(simulated_price)
                    simulated_daily_prices.append((date, weekday, simulated_price))
                    break

            except KeyError as e:
                print(f"KeyError during simulation: {e}")
                raise

            retry_count += 1

        if retry_count >= max_retries:
            print(f"Warning: Maximum retries reached for {date}. Using last valid price.")
            simulated_daily_prices.append((date, weekday, simulated_daily_prices[-1][2]))

    # Adjust daily prices to ensure their average matches the monthly price
    for month, prices in daily_prices_by_month.items():
        month_avg_price = monthly_simulated_prices.loc[
            monthly_simulated_prices['Month'] == month, 'Simulated_Price'
        ].values[0]
        adjustment_factor = month_avg_price / np.mean(prices)
        for i, (date, weekday, price) in enumerate(simulated_daily_prices):
            if date.strftime("%B") == month:
                simulated_daily_prices[i] = (date, weekday, price * adjustment_factor)

    # Create DataFrame for results
    simulated_daily_df = pd.DataFrame(simulated_daily_prices, columns=['Date', 'Weekday', 'Simulated_Price'])

    return simulated_daily_df


# Example usage
simulated_daily_prices = simulate_daily_prices_with_pacf_sampling(
    monthly_simulated_prices=simulated_monthly_prices,
    overall_weekly_seasonality=overall_weekly_seasonality,
    significant_lags_vertical=significant_lags_vertical,
    weekday_results_df=weekday_results_df
)

print(simulated_daily_prices)
trigonometric_horizontal_seasonality_data
'''
#-----------------------------------------------------------------------------#
#                          Simulating Hourly Prices                           # 
#-----------------------------------------------------------------------------#
'''
def simulate_hourly_prices_with_acf_pacf_sampling(
    daily_simulated_prices, trigonometric_seasonality_data, significant_lags_horizontal, hourly_results_df
):
    """
    Simulates hourly electricity prices using Monte Carlo simulation with trigonometric seasonality
    and both ACF and PACF values.
    """
    import numpy as np
    import pandas as pd
    import ast
    import calendar

    # Debug: Ensure hourly_results_df contains the correct data
    #print("hourly_results_df columns:", hourly_results_df.columns)
    #print("hourly_results_df sample:", hourly_results_df.head())

    # Prepare seasonality data
    seasonality_by_day = trigonometric_seasonality_data.set_index("Day").to_dict(orient="index")

    # Extract PACF and ACF values dynamically for hourly data
    pacf_lags = significant_lags_horizontal[significant_lags_horizontal['Type'] == 'PACF']
    acf_lags_df = significant_lags_horizontal[significant_lags_horizontal['Type'] == 'ACF']

    # Extract error distributions for each hour
    empirical_errors = {}
    for _, row in hourly_results_df.iterrows():
        try:
            dist_name = row['Best_Distribution']
            params = row['Parameters']
            if isinstance(params, str):
                params = ast.literal_eval(params)
            dist = globals()[dist_name]
            empirical_errors[row['Hour']] = (dist, params)
        except KeyError as e:
            print(f"KeyError during empirical_errors generation: {e}. Check 'Hour' column.")
            raise

    # Initialize simulation
    simulated_hourly_prices = []
    for _, daily_row in daily_simulated_prices.iterrows():
        date = daily_row['Date']
        weekday_index = pd.Timestamp(date).weekday()
        day_name = calendar.day_name[weekday_index]
        avg_daily_price = daily_row['Simulated_Price']
        hourly_prices = np.zeros(24)  # 24 hours per day

        for hour in range(24):
            try:
                # Seasonal components
                seasonal_component = (
                    seasonality_by_day.get(day_name, {}).get('Day_s1 (sin 2πt_day)', 0) * np.sin(2 * np.pi * (hour / 12 if hour < 12 else 0)) +
                    seasonality_by_day.get(day_name, {}).get('Day_s2 (cos 2πt_day)', 0) * np.cos(2 * np.pi * (hour / 12 if hour < 12 else 0)) +
                    seasonality_by_day.get(day_name, {}).get('Night_s1 (sin 2πt_night)', 0) * np.sin(2 * np.pi * ((hour - 12) / 12 if hour >= 12 else 0)) +
                    seasonality_by_day.get(day_name, {}).get('Night_s2 (cos 2πt_night)', 0) * np.cos(2 * np.pi * ((hour - 12) / 12 if hour >= 12 else 0)) +
                    seasonality_by_day.get(day_name, {}).get('Annual_s1 (sin 2πt_year)', 0) * np.sin(2 * np.pi * (hour / 24)) +
                    seasonality_by_day.get(day_name, {}).get('Annual_s2 (cos 2πt_year)', 0) * np.cos(2 * np.pi * (hour / 24)) +
                    seasonality_by_day.get(day_name, {}).get('Annual_Intercept', 0)
                )

                # PACF contribution
                pacf_contribution = 0
                for _, pacf_row in pacf_lags[pacf_lags['Column'] == f"{day_name}_diff1"].iterrows():
                    lag = pacf_row['Lag']
                    pacf_value = pacf_row['Value']
                    adjusted_hour = hour - lag
                    if adjusted_hour >= 0:
                        pacf_contribution += pacf_value * (hourly_prices[adjusted_hour] - seasonal_component)

                # ACF contribution
                acf_contribution = 0
                for _, acf_row in acf_lags_df[acf_lags_df['Column'] == f"{day_name}_diff1"].iterrows():
                    lag = acf_row['Lag']
                    acf_value = acf_row['Value']
                    adjusted_hour = hour - lag
                    if adjusted_hour >= 0:
                        acf_contribution += acf_value * hourly_prices[adjusted_hour]

                simulated_price = seasonal_component + pacf_contribution + acf_contribution

                # Add stochastic error
                dist, params = empirical_errors.get(hour, (None, None))
                if dist:
                    error = dist.rvs(*params, size=1)[0]
                    simulated_price += error

                # Ensure non-negative prices
                hourly_prices[hour] = max(simulated_price, 0)

            except KeyError as e:
                print(f"KeyError during simulation for hour {hour}: {e}.")
                raise

        # Normalize hourly prices to match daily average price
        hourly_mean = np.mean(hourly_prices)
        if hourly_mean > 0:
            hourly_prices *= (avg_daily_price / hourly_mean)

        daily_hours = [(date, day_name, h, hourly_prices[h]) for h in range(24)]
        simulated_hourly_prices.extend(daily_hours)

    # Create DataFrame for results
    simulated_hourly_df = pd.DataFrame(simulated_hourly_prices, columns=['Date', 'Weekday', 'Hour', 'Simulated_Price'])

    return simulated_hourly_df



# Example Usage
simulated_hourly_prices = simulate_hourly_prices_with_acf_pacf_sampling(
    daily_simulated_prices=simulated_daily_prices,
    trigonometric_seasonality_data=trigonometric_horizontal_seasonality_data,
    significant_lags_horizontal=significant_lags_horizontal,
    hourly_results_df=hourly_results_df
)

print(simulated_hourly_prices)

'''
#-----------------------------------------------------------------------------#
#                 Full Simulation and Statistical Analysis                     # 
#-----------------------------------------------------------------------------#
'''

def run_full_monte_carlo_simulation(
    monthly_seasonality, significant_lags_monthly, monthly_results_df,
    overall_weekly_seasonality, significant_lags_vertical, weekday_results_df,
    trigonometric_seasonality_data, significant_lags_horizontal, hourly_results_df,
    num_trials=5
):
    """
    Executes the full Monte Carlo simulation process (monthly, daily, hourly) 10,000 times,
    applying scaling and mean adjustments only once for each level.

    Parameters:
        monthly_seasonality: DataFrame
        significant_lags_monthly: DataFrame
        monthly_results_df: DataFrame
        overall_weekly_seasonality: DataFrame
        significant_lags_vertical: DataFrame
        weekday_results_df: DataFrame
        trigonometric_seasonality_data: DataFrame
        significant_lags_horizontal: DataFrame
        hourly_results_df: DataFrame
        num_trials: int (default=10000)

    Returns:
        tuple: Lists of monthly, daily, and hourly simulation results.
    """
    import pandas as pd

    # Apply variations for each level
    print("Applying variations to monthly price simulation...")
    adjusted_monthly_results_df = apply_variation_with_console(monthly_results_df)

    print("Applying variations to daily price simulation...")
    adjusted_weekday_results_df = apply_variation_with_console(weekday_results_df)

    print("Applying variations to hourly price simulation...")
    adjusted_hourly_results_df = apply_variation_with_console(hourly_results_df)

    # Storage for simulation results
    monthly_simulations = []
    daily_simulations = []
    hourly_simulations = []

    # Run simulations for the specified number of trials
    for trial in range(num_trials):
        print(f"Running simulation {trial + 1}/{num_trials}...")

        # Monthly simulation
        simulated_monthly_prices = simulate_monthly_prices_with_pacf(
            monthly_seasonality=monthly_seasonality,
            significant_lags_monthly=significant_lags_monthly,
            monthly_results_df=adjusted_monthly_results_df,
            num_months=12
        )
        monthly_simulations.append(simulated_monthly_prices)

        # Daily simulation
        try:
            simulated_daily_prices = simulate_daily_prices_with_pacf_sampling(
                monthly_simulated_prices=simulated_monthly_prices,
                overall_weekly_seasonality=overall_weekly_seasonality,
                significant_lags_vertical=significant_lags_vertical,
                weekday_results_df=adjusted_weekday_results_df
            )
            daily_simulations.append(simulated_daily_prices)
        except KeyError as e:
            print(f"Error during daily simulation: {e}. Check 'Weekday' column in 'weekday_results_df'.")
            raise

        # Hourly simulation
        try:
            simulated_hourly_prices = simulate_hourly_prices_with_acf_pacf_sampling(
                daily_simulated_prices=simulated_daily_prices,
                trigonometric_seasonality_data=trigonometric_seasonality_data,
                significant_lags_horizontal=significant_lags_horizontal,
                hourly_results_df=adjusted_hourly_results_df
            )
            hourly_simulations.append(simulated_hourly_prices)
        except KeyError as e:
            print(f"Error during hourly simulation: {e}. Check 'Hour' column in 'hourly_results_df'.")
            raise

    return monthly_simulations, daily_simulations, hourly_simulations

# Example Usage:
# Replace placeholders below with actual DataFrame objects
monthly_results, daily_results, hourly_results = run_full_monte_carlo_simulation(
     monthly_seasonality=monthly_seasonality,
     significant_lags_monthly=significant_lags_monthly,
     monthly_results_df=monthly_results_df,
     overall_weekly_seasonality=overall_weekly_seasonality,
     significant_lags_vertical=significant_lags_vertical,
     weekday_results_df=weekday_results_df,
     trigonometric_seasonality_data=trigonometric_horizontal_seasonality_data,
     significant_lags_horizontal=significant_lags_horizontal,
     hourly_results_df=hourly_results_df
)

print("Monthly Results:", len(monthly_results))
print("Daily Results:", len(daily_results))
print("Hourly Results:", len(hourly_results))


###############################################################################
###############################################################################

# Define error metric functions
def calculate_mae(actual, simulated):
    return np.mean(np.abs(actual - simulated))

def calculate_mape(actual, simulated):
    return np.mean(np.abs((actual - simulated) / actual)) * 100

def calculate_rmse(actual, simulated):
    return np.sqrt(np.mean((actual - simulated) ** 2))

# Use preprocessed dataframes directly
aggregated_monthly_data = aggregated_monthly_data
vertical_daily_aggregates = vertical_daily_aggregates
hourly_data = aggregated_horizontal_hourly_data

# Example list of simulated DataFrames
simulated_monthly_results = [pd.DataFrame({'Simulated': np.random.rand(12)}) for _ in range(3)]
simulated_daily_results = [pd.DataFrame({'Simulated': np.random.rand(365)}) for _ in range(3)]
simulated_hourly_results = [pd.DataFrame({'Simulated': np.random.rand(8760)}) for _ in range(3)]

# Error metrics storage
monthly_errors = []
daily_errors = []
hourly_errors = []

# Compute metrics for monthly data
for sim_df in simulated_monthly_results:
    actual = aggregated_monthly_data['Average_MEP']
    simulated = sim_df['Simulated'][:len(actual)]  # Truncate to match length
    mae = calculate_mae(actual, simulated)
    mape = calculate_mape(actual, simulated)
    rmse = calculate_rmse(actual, simulated)
    monthly_errors.append({'MAE': mae, 'MAPE': mape, 'RMSE': rmse})

# Compute metrics for daily data
for sim_df in simulated_daily_results:
    actual = vertical_daily_aggregates['Average_MEP']
    simulated = sim_df['Simulated'][:len(actual)]  # Truncate to match length
    mae = calculate_mae(actual, simulated)
    mape = calculate_mape(actual, simulated)
    rmse = calculate_rmse(actual, simulated)
    daily_errors.append({'MAE': mae, 'MAPE': mape, 'RMSE': rmse})

# Compute metrics for hourly data
for sim_df in simulated_hourly_results:
    actual = hourly_data['Average_MEP']
    simulated = sim_df['Simulated'][:len(actual)]  # Truncate to match length
    mae = calculate_mae(actual, simulated)
    mape = calculate_mape(actual, simulated)
    rmse = calculate_rmse(actual, simulated)
    hourly_errors.append({'MAE': mae, 'MAPE': mape, 'RMSE': rmse})

# Convert results to DataFrames
monthly_errors_df = pd.DataFrame(monthly_errors)
daily_errors_df = pd.DataFrame(daily_errors)
hourly_errors_df = pd.DataFrame(hourly_errors)

# Compute the average error metrics
monthly_avg_errors = monthly_errors_df.mean()
daily_avg_errors = daily_errors_df.mean()
hourly_avg_errors = hourly_errors_df.mean()

# Display the average error metrics
print("Average Monthly Errors:")
print(monthly_avg_errors)
print("\nAverage Daily Errors:")
print(daily_avg_errors)
print("\nAverage Hourly Errors:")
print(hourly_avg_errors)

# Visualization of average error metrics
def plot_average_errors(avg_errors, title):
    avg_errors.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange', 'green'], alpha=0.7)
    plt.title(title)
    plt.xlabel('Error Metric')
    plt.ylabel('Average Value')
    plt.grid(axis='y')
    plt.show()

# Plot average error metrics
plot_average_errors(monthly_avg_errors, "Average Monthly Error Metrics")
plot_average_errors(daily_avg_errors, "Average Daily Error Metrics")
plot_average_errors(hourly_avg_errors, "Average Hourly Error Metrics")


###############################################################################
###############################################################################


# Visualization: Heatmap of Errors

def plot_error_heatmap(error_df, title):
    """
    Plot a heatmap of error metrics to visualize patterns across simulations.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        error_df.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5
    )
    plt.title(title)
    plt.xlabel("Simulation Index")
    plt.ylabel("Error Metric")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.show()

# Call the function for each error DataFrame
plot_error_heatmap(monthly_errors_df, "Heatmap of Monthly Errors")
plot_error_heatmap(daily_errors_df, "Heatmap of Daily Errors")
plot_error_heatmap(hourly_errors_df, "Heatmap of Hourly Errors")

# Ranking Simulations Based on Composite Score

def rank_simulations(error_df):
    """
    Rank simulations based on a composite error score (weighted sum of MAE, MAPE, and RMSE).
    """
    # Assign weights to error metrics
    weights = {
        "MAE": 0.4,  # Adjust as per importance
        "MAPE": 0.3,
        "RMSE": 0.3
    }

    # Compute weighted score
    error_df["Composite_Score"] = (
        weights["MAE"] * error_df["MAE"] +
        weights["MAPE"] * error_df["MAPE"] +
        weights["RMSE"] * error_df["RMSE"]
    )

    # Rank simulations (lower score is better)
    error_df["Rank"] = error_df["Composite_Score"].rank(method="min")

    return error_df.sort_values(by="Rank")

# Rank and display the top simulations for each level
ranked_monthly_errors = rank_simulations(monthly_errors_df)
ranked_daily_errors = rank_simulations(daily_errors_df)
ranked_hourly_errors = rank_simulations(hourly_errors_df)

print("Top Monthly Simulations:")
print(ranked_monthly_errors.head())
print("\nTop Daily Simulations:")
print(ranked_daily_errors.head())
print("\nTop Hourly Simulations:")
print(ranked_hourly_errors.head())

# Visualization: Composite Score Bar Chart

def plot_composite_score_bar(error_df, title):
    """
    Plot a bar chart of composite scores for all simulations.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=error_df.index, y=error_df["Composite_Score"], palette="viridis")
    plt.title(title)
    plt.xlabel("Simulation Index")
    plt.ylabel("Composite Score")
    plt.xticks(rotation=90)
    plt.grid(axis="y")
    plt.show()

# Call the function for each level
plot_composite_score_bar(ranked_monthly_errors, "Composite Scores for Monthly Simulations")
plot_composite_score_bar(ranked_daily_errors, "Composite Scores for Daily Simulations")
plot_composite_score_bar(ranked_hourly_errors, "Composite Scores for Hourly Simulations")

###############################################################################
###############################################################################

