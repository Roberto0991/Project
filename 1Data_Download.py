#Code to download the data and send notifications on the current status in a chosen country

import pandas as pd
from win10toast import ToastNotifier

url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv' #https://github.com/owid/covid-19-data/tree/master/public/data
data = pd.read_csv(url,index_col=0, usecols = [0,1,2,3,4,5,6,7,8,9,10], parse_dates=[1]) #Columns 0 to 10 selected only
data.to_csv('corona.csv') #Save the file

#Input variables for alert
Location = 'Switzerland' #Select the country for alerts
notification_duration = 5

country = data.loc[data['location'] == Location, 'location':'new_deaths_per_million']
latestdate = country['date'].max()
country.set_index('date', inplace =True)
update = country.loc[latestdate] #Selecting the latest data for update

message = "Country = {}, Total Cases = {}, New Cases = {}, Total Deaths = {}, New Deaths = {}, Total Cases per million = {}".format(*update)   
toaster = ToastNotifier()
toaster.show_toast("Coronavirus Update - " + latestdate, message, duration = notification_duration)

print(country.index)
