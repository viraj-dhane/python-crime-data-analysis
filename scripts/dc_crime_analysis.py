# Importing  the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Import data
df1 = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\Crime_Incidents_in_2012.csv")
df2 = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\Crime_Incidents_in_2013.csv")
df3 = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\Crime_Incidents_in_2014.csv")
df4 = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\Crime_Incidents_in_2015.csv")
df5 = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\Crime_Incidents_in_2016.csv")
df6 = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\Crime_Incidents_in_2017.csv")
df7 = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\Crime_Incidents_in_2018.csv")

# Creating a data frame with column names
COLUMN_NAMES = ['X', 'Y', 'CCN', 'REPORT_DAT', 'SHIFT', 'METHOD', 'OFFENSE', 'BLOCK', 'XBLOCK',
                 'YBLOCK', 'WARD', 'ANC', 'DISTRICT', 'PSA','NEIGHBORHOOD_CLUSTER', 'BLOCK_GROUP', 'CENSUS_TRACT',
                 'VOTING_PRECINCT', 'LATITUDE', 'LONGITUDE', 'BID', 'START_DATE','END_DATE', 'OBJECTID']
df = pd.DataFrame(columns=COLUMN_NAMES)

# Check column names in all data frames
print(df.columns)
print(df1.columns)
print(df2.columns)
print(df3.columns)
print(df4.columns)
print(df5.columns)
print(df6.columns)
print(df7.columns)

# Concatinate all data into single data frame df
df = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index= True)
#print(df.info())
print(df)   

# Checking the null values in each column of df and sorting them as percentage of total rows in data frame
print(df.isnull().sum().sort_values(ascending=False))
print(((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False))

print("---***<<< Crimes commited per Duty Shift: variable = 'SHIFT' >>>***---")
var_count = df.groupby('SHIFT')
print(var_count['CCN'].count())

print("---==< Crimes committed per police district: variable = 'DISTRICT' >==---")
var_count = df.groupby('DISTRICT')
print(var_count.CCN.count())

print("---==< Crimes|OFFENSES committed per BLOCK: variable = 'BLOCK' >==---")
print(df.groupby('BLOCK').OFFENSE.value_counts())

# Removing column BLOCK from data frame
#print (np.count_nonzero(df['BLOCK'].unique()))
#print (np.count_nonzero(df['BLOCK_GROUP'].unique()))
df.drop('BLOCK', axis=1, inplace=True)
df.drop('BLOCK_GROUP', axis=1, inplace=True)
#print(df.head())
print('***************************************************')

# Handeling missing values
# If END_DATE is NaN, then use START_DATE
df['END_DATE'].fillna(df['START_DATE'], inplace=True)

# If VOTING_PRECINCT is NaN, then set it to 0
df['VOTING_PRECINCT'].fillna(0, inplace=True)

# If NEIGHBORHOOD_CLUSTER is NaN, then set it to 0
df['NEIGHBORHOOD_CLUSTER'].fillna(0, inplace=True)

# If CENSUS_TRACT is NaN, then set it to 0
df['CENSUS_TRACT'].fillna(0, inplace=True)

# If WARD is NaN, then set it to 0
df['WARD'].fillna(0, inplace=True)

"""
To handle missing values in PSA column:
- Create a DataFrame that holds the central location of each Police Service Area (PSA).
- According to the information from the DC metropolitan information and OpenData DC, PSAs are smaller than Police Districts. Therefore, identifying the associated PSA will provide better accuracy.
- The PSA contains the District, so the District can be imputed from the PSA.
"""
psa_loc = pd.DataFrame(df[['PSA','XBLOCK','YBLOCK']].groupby('PSA').median())
#  ---==< Estimate PSA based on proximity to each area's centroid >==---
def nearbyPSA(nPSA,dX,dY):
    # Default to the current PSA ID
    nearbyPSA = nPSA
    
    # Only operate on missing IDs
    if (pd.isnull(nPSA)):
        minDist = 9e99  # Set the initial closest distance to a large value
        nearbyPSA = 0
        
        for PSA_ID, PSA in psa_loc.iterrows():
            # Calculate the distance between the report and the current PSA using the Eucleadian distance
            thisDist = math.sqrt((dX - PSA['XBLOCK'])**2 + (dY - PSA['YBLOCK'])**2)
            
            # If this distance is smaller than the current minimum distance, update the minimum distance
            if (thisDist < minDist):
                minDist = thisDist # Replace the minimum distance with the current distance
                nearbyPSA = PSA_ID # To assign the PSA this is related to
                
    # Return the ID for the closest PSA
    return [nearbyPSA, int(nearbyPSA / 100)]

df['PSA_ID'] = 0
df['DistrictID'] = 0
df[['PSA_ID','DistrictID']] = list(map(nearbyPSA,df['PSA'],df['XBLOCK'],df['YBLOCK']))
#print(df[['PSA','DISTRICT','PSA_ID','DistrictID']][df['PSA'].isnull()])

# Check if data frame has duplicate values
print(df.duplicated().sum())

# Strip 'Precinct ' from VOTING_PRECINCT values
df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].apply(str).map(lambda x: x.lstrip('Precinct '))

# Strip 'Cluster ' from NEIGHBORHOOD_CLUSTER values
df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].apply(str).map(lambda x: x.lstrip('Cluster '))

# Convert REPORT_DAT to datetime
#df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'])

# Convert OFFENSE to numeric codes
offense_mapping = {'theft/other':1, 'theft f/auto':2, 'burglary':3, 'assault w/dangerous weapon':4, 'robbery':5,
                  'motor vehicle theft':6, 'homicide':7, 'sex abuse':8, 'arson':9}
df['OFFENSE_Code'] = df['OFFENSE'].str.lower().map(offense_mapping).astype('category')
df['OFFENSE'] = df['OFFENSE'].str.replace('DANGEROUS WEAPON', 'DW')

# Convert METHOD to numeric codes
method_mapping = {'others':1, 'gun':2, 'knife':3}
df['METHOD_Code'] = df['METHOD'].str.lower().map(method_mapping).astype('category')

# Convert DISTRICT to integer data type
df['DistrictID'] = df['DistrictID'].astype(np.int64)

# Convert PSA to integer data type
df['PSA_ID'] = df['PSA_ID'].astype(np.int64)

# Convert WARD to integer data type
df['WARD'] = df['WARD'].astype(np.int64)

# Convert ANC to numeric codes
anc_mapping = {'1A':11, '1B':12, '1C':13, '1D':14,   
               '2A':21, '2B':22, '2C':23, '2D':24, '2E':25, '2F':26,
               '3B':32, '3C':33, '3D':34, '3E':35, '3F':36, '3G':37,
               '4A':41, '4B':42, '4C':43, '4D':44,
               '5A':51, '5B':52, '5C':53, '5D':54, '5E':55,
               '6A':61, '6B':62, '6C':63, '6D':64, '6E':65,
               '7B':72, '7C':73, '7D':74, '7E':75, '7F':76,
               '8A':81, '8B':82, '8C':83, '8D':84, '8E':85}
df['ANC'] = df['ANC'].map(anc_mapping).astype('category')

# Convert NEIGHBORHOOD_CLUSTER to integer data type 
df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].astype(np.int64)

# Convert CENSUS_TRACT to integer data type
df['CENSUS_TRACT'] = df['CENSUS_TRACT'].astype(np.int64)

# Convert VOTING_PRECINCT to integer data type
df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].astype(np.int64)

# Convert CCN to integer data type
df['CCN'] = df['CCN'].astype(np.int64)

# Convert XBLOCK, YBLOCK to float data type
df['XBLOCK'] = df['XBLOCK'].astype(np.float64)
df['YBLOCK'] = df['YBLOCK'].astype(np.float64)

# Convert START_DATE, END_DATE to dateime
df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='ISO8601', errors='coerce')
df['END_DATE'] = pd.to_datetime(df['END_DATE'], format='ISO8601', errors='coerce')

# Creating feature for crime type 1 = Violent, 2 = Non-violent
violent_offense = [4, 5, 7, 8]
df['CRIME_TYPE'] = np.where(df['OFFENSE_Code'].isin(violent_offense), 1, 2)
df['CRIME_TYPE'] = df['CRIME_TYPE'].astype('category')

# Calculate the age of crime END_DATE - START_DATE in seconds
df['AGE'] = (df['END_DATE'] - df['START_DATE'])/np.timedelta64(1, 's')

"""
Create a new feature TIME_TO_REPORT to indicate the timespan between the latest time crime was commited and the time it was repported to MPD.
i.e. time taken to report the crime, REPORT_DAT - END_DATE in seconds 
"""
df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'], format='ISO8601', errors='coerce')
df['TIME_TO_REPORT'] = (df['REPORT_DAT'] - df['END_DATE'])/np.timedelta64(1, 's')

df['DATE'] = pd.to_datetime(df['END_DATE'], format = '%d/%m/%Y %H:%M:%S')

"""
Created additional columns that help analyze the timing of the crimes,
including day, week, month, year, as well as quarters, weekdays, and weekends.
"""
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month
df['Day'] = df['DATE'].dt.day
df['hour'] = df['DATE'].dt.hour
df['dayofyear'] = df['DATE'].dt.dayofyear
df['week'] = df['DATE'].apply(lambda x: (x.day - 1) // 7 + 1) # Calculate the week of the month
df['weekofyear'] = df['DATE'].dt.isocalendar().week
df['dayofweek'] = df['DATE'].dt.dayofweek
df['weekday'] = df['DATE'].dt.weekday
df['quarter'] = df['DATE'].dt.quarter

#Dealing with outliers
print(df['START_DATE'][df['START_DATE']<'1/1/2012'].count())
df = df[(df['Year'] >= 2012) & (df['Year'] <= 2018)]

df.loc[:, 'Year'] = df['Year'].astype(int)
df.loc[:, 'Month'] = df['Month'].astype(int)
df.loc[:, 'Day'] = df['Day'].astype(int)
df.loc[:, 'hour'] = df['hour'].astype(int)
df.loc[:, 'dayofyear'] = df['dayofyear'].astype(int)
df.loc[:, 'week'] = df['week'].astype(int)
df.loc[:, 'weekofyear'] = df['weekofyear'].astype(int)
df.loc[:, 'dayofweek'] = df['dayofweek'].astype(int)
df.loc[:, 'weekday'] = df['weekday'].astype(int)
df.loc[:, 'quarter'] = df['quarter'].astype(int)

# Storing the clean data set in a file named DC_Crime_Data
df.to_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\DC_Crime_Clean_Data.csv")

print("**-----<<< Data cleaning complete >>>-----**")

df = pd.read_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\Data Sets\DC_Crime_Clean_Data.csv")

print("---***<<< Crimes commited per Duty Shift: variable = 'SHIFT' >>>***---")
var_count = df.groupby('SHIFT')
print(var_count['CCN'].count())

print("---==< Crimes committed per police district: variable = 'DISTRICT' >==---")
var_count = df.groupby('DISTRICT')
print(var_count.CCN.count())

print('---<Examine the frequency of types of crimes/offenses>--======')
print(' Total Offenses - Count')
total_crime = df.CCN.count()
print (total_crime)
crime_rate = df.groupby('OFFENSE')
print (' Offense Type - Count')
print (crime_rate.CCN.count())
print (' Offense as percentage of total ')
print ((crime_rate.CCN.count() / total_crime) * 100.0)
print()
print (' Offense Rate per 100,000 ')
print (crime_rate.CCN.count() / 6.93972 )
print (' Odds of being a victim - by offense')
print (crime_rate.CCN.count() / 693972)

print()

print('-<Examine the frequency of types of crimes/offenses by Method>===')
print(' Total Methods - Count') 
total_crime = df.CCN.count()
print (total_crime)
method_rate = df.groupby('METHOD')
print ('Method Type - Count')
print(method_rate.CCN.count())
print (' Method as percentage of total ')
print ((method_rate.CCN.count() / total_crime) * 100.0)
print()
print (' Method Rate per 100,000 ')
print( method_rate.CCN.count() / 6.93972)
print (' Odds of being a victim - by method')
print (method_rate.CCN.count() / 693972)

print()
print('***************************************************')
print()

print('---< Examine the age of crimes/offense by Time >---')
temp = df['AGE'] / 3600 #hours
print( temp.describe())
# excluding temp two std away from mean
print (temp[~(np.abs(temp - temp.mean())>(2*temp.std()))].describe())

print('***************************************************')

print('---< Examine the frequency of time it takes to report a of crimes/offenses >---')
temp = df['TIME_TO_REPORT'] / 3600  #hours
print(temp.describe())
# excluding temp two std away from mean
print (temp[~(np.abs(temp - temp.mean())>(2*temp.std()))].describe())

print('***************************************************')

"""
-----<<< DATA VISUALIZATION >>>----- 
"""

# Setting the visual
sns.set(font_scale=2)
cmap = sns.diverging_palette(220, 10, as_cmap=True) # one of the many color mappings
sns.set_style('whitegrid')

print("---< Categorized  Offences by Shift  >----")
temp_var = df[['SHIFT', 'CRIME_TYPE']]
plt.figure(figsize=(8, 5))
count_plot = sns.countplot(x='SHIFT', hue='CRIME_TYPE', data=temp_var) # crime type 1 = Violent, 2 = Non-violent
count_plot.set(ylabel='Count')
plt.show()
#Box plot
box_plot = sns.boxplot(x='SHIFT', y='hour', data=df, palette='winter_r')
box_plot.set(ylabel= 'Hour')
plt.show()
# Pie chart
labels = df['SHIFT'].value_counts().index
fig, ax = plt.subplots()
ax.pie(df.SHIFT.value_counts(), labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

print("---< Categorized  Offences at District Level >----")
temp_var = df[['DistrictID', 'METHOD']]
temp_var1 = df[['DistrictID', 'METHOD']]
temp_var1 = temp_var1[temp_var1['METHOD'] != "OTHERS"]
plt.figure(figsize=(10, 5))
count_plot = sns.countplot(x='DistrictID', hue='METHOD', data=temp_var) #count plot for methods - Gun, Knife, Others
count_plot.set(ylabel='Count')
plt.show()

plt.figure(figsize=(10, 5))
count_plot = sns.countplot(x='DistrictID', hue='METHOD', data=temp_var1) #count plot for methods - Gun, Knife
count_plot.set(ylabel='Count')
plt.show()

temp_var = df[['DistrictID', 'CRIME_TYPE']]
plt.figure(figsize=(15, 9))
count_plot = sns.countplot(x='DistrictID', hue='CRIME_TYPE', data=temp_var)
count_plot.set(ylabel='Count')
plt.show()

temp_var = pd.crosstab(df.DistrictID, df.OFFENSE)
stacked_bar_plot = temp_var.plot(kind='bar', stacked=True, figsize=(25, 15)) # offence type 1 = theft/other, 2 = theft f/auto, 3 = burglary, 4 = assault w/dangerous weapon, 5 = robbery, 6 = motor vehicle theft, 7 = homicide, 8 = sex abuse, 9 = arson
stacked_bar_plot.set(ylabel='Count')
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.085), ncol=9, fontsize='medium')
plt.show()

print("---< Categorized  Offences Week, Month, Quarter of Year >----")
temp_var = df[['Month', 'CRIME_TYPE']]
plt.figure(figsize=(20, 10))
count_plot = sns.countplot(x='Month', hue='CRIME_TYPE', data=temp_var)
count_plot.set(ylabel='Count')
plt.show()

temp_var = df[['weekofyear', 'CRIME_TYPE']]
plt.figure(figsize=(50, 45))
count_plot = sns.countplot(x= 'weekofyear', hue='CRIME_TYPE', data=temp_var)
count_plot.set(xlabel='Week of Year')
count_plot.set(ylabel='Count')
count_plot.tick_params(axis='x', labelsize=10)
count_plot.tick_params(axis='y', labelsize=10)
plt.show()

temp_var = df[['quarter', 'CRIME_TYPE']]
plt.figure(figsize=(50, 45))
count_plot = sns.countplot(x= 'quarter', hue='CRIME_TYPE', data=temp_var)
count_plot.set(ylabel='Count')
plt.show()

# Function to convert counts per category into percentages
def percentConv(x):
    return x / float(sum(x))

print(pd.crosstab(df.OFFENSE, df.SHIFT, margins=True))
print(pd.crosstab(df.OFFENSE, df.SHIFT).apply(percentConv, axis=1))

print (pd.crosstab(df.DistrictID, df.CRIME_TYPE, margins=True))
print (pd.crosstab(df.DistrictID, df.CRIME_TYPE).apply(percentConv, axis=1))

print (pd.crosstab(df.OFFENSE, df.METHOD, margins=True))
print(pd.crosstab(df.OFFENSE, df.METHOD).apply(percentConv, axis=1))

day_of_week = df.dayofweek  # 0 = Monday ... 6 = Sunday
print( pd.crosstab(df.OFFENSE, day_of_week, colnames=['dayofweek'], margins=True))
print(pd.crosstab(df.OFFENSE, day_of_week, colnames=['dayofweek']).apply(percentConv, axis=1))

print ('Percentage of Crime in each District')
temp_var_total = df.DistrictID.value_counts().sum()
print(df.DistrictID.value_counts() / temp_var_total)

# Analysis of Crime Reporting Time
print("*************************************************** Reporting times per offense type ***************************************************")
plt.figure(figsize=(30,10))  #set up a wide chart for better visual and understanding
sns.set(font_scale=1.25)
# Create a subset of data having response time from 0 to 24 hours (Time taken to report crime)
plt_test = df[df.TIME_TO_REPORT < 86400][df.TIME_TO_REPORT > 0]
# Create the box plot - report the response time in hours instead of seconds. Group by Offense, and color by Shift
box_plot = sns.boxplot(x=plt_test.OFFENSE, y=plt_test.TIME_TO_REPORT/3600.0, hue=plt_test.SHIFT)
box_plot.set(ylabel='Time To Report (Hour)')
plt.legend(loc='upper right')   #set location of legend to upper right corner
plt.show()

print("*************************************************** Reporting times when a Dangerous Weapon is involved ***************************************************")
plt.figure(figsize=(30,10)) #set up a wide chart for better visual and understanding
sns.set(font_scale=2)
# Create a subset of data having response time from 0 to 24 hours (Time taken to report crime)
plt_test = df[df.TIME_TO_REPORT < 86400][df.TIME_TO_REPORT > 0]
# Create the box plot - report the response time in hours instead of seconds.  Group by Method, and color by Shift
box_plot = sns.boxplot(x=plt_test.METHOD, y=plt_test.TIME_TO_REPORT/3600.0, hue=plt_test.SHIFT)
box_plot.set(ylabel='Time To Report (Hour)')
plt.legend(loc='upper right')   #set location of legend to upper right corner
plt.show()

print("*************************************************** Reporting times per District and Shift ***************************************************")
plt.figure(figsize=(40,10))
sns.set(font_scale=3)
# Create a subset of data having response time from 0 to 24 hours (Time taken to report crime)
plt_test = df[df.TIME_TO_REPORT < 86400][df.TIME_TO_REPORT > 0]
# Create the box plot - report the response time in hours instead of seconds. Group by Method, and color by Shift
box_plot = sns.boxplot(x=plt_test.DistrictID, y=plt_test.TIME_TO_REPORT/3600.0, hue=plt_test.SHIFT)
box_plot.set(ylabel='Time To Report (Hour)')
plt.legend(loc='upper right')   #set location of legend to upper right corner
plt.show()

print("*************************************************** Plot of Violent Crimes ***************************************************")
plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ROBBERY'], df['LATITUDE'][df['OFFENSE']=='ROBBERY'], s=50, alpha=0.3, color=[0.0,1.0,1.0], lw=0, label='Robbery')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ASSAULT W/DW'], df['LATITUDE'][df['OFFENSE']=='ASSAULT W/DW'], s=50, alpha=0.5, color='g', lw=0, label='Assault')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='SEX ABUSE'], df['LATITUDE'][df['OFFENSE']=='SEX ABUSE'], s=50, alpha=0.3, color='b', lw=0, label='Sex Abuse')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='HOMICIDE'], df['LATITUDE'][df['OFFENSE']=='HOMICIDE'], s=50, alpha=0.3, color='r', lw=0, label='Homicide')
plt.scatter(-77.03654,38.89722, s=70, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968, s=70, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right')
plt.show()

print("*************************************************** Plot of Violent Crime - Robbery ***************************************************")
plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ROBBERY'], df['LATITUDE'][df['OFFENSE']=='ROBBERY'], s=50, alpha=0.3, color=[0.0,1.0,1.0], lw=0, label='Robbery')
plt.scatter(-77.03654,38.89722, s=70, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968, s=70, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right')
plt.show()

print("*************************************************** Plot of Violent Crime - Assault ***************************************************")
plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ASSAULT W/DW'], df['LATITUDE'][df['OFFENSE']=='ASSAULT W/DW'], s=50, alpha=0.5, color='g', lw=0, label='Assault')
plt.scatter(-77.03654,38.89722, s=70, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968, s=70, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right')
plt.show()

print("*************************************************** Plot of Violent Crime - Sex Abuse ***************************************************")
plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='SEX ABUSE'], df['LATITUDE'][df['OFFENSE']=='SEX ABUSE'], s=50, alpha=0.3, color='b', lw=0, label='Sex Abuse')
plt.scatter(-77.03654,38.89722, s=70, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968, s=70, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right')
plt.show()

print("*************************************************** Plot of Violent Crime - Homicide ***************************************************")
plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='HOMICIDE'], df['LATITUDE'][df['OFFENSE']=='HOMICIDE'], s=50, alpha=0.3, color='r', lw=0, label='Homicide')
plt.scatter(-77.03654,38.89722, s=70, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968, s=70, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right')
plt.show()

print("*************************************************** Plot of Violent Crimes - Sex Abuse and Homicide ***************************************************")
plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='SEX ABUSE'], df['LATITUDE'][df['OFFENSE']=='SEX ABUSE'], s=50, alpha=0.3, color='b', lw=0, label='Sex Abuse')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='HOMICIDE'], df['LATITUDE'][df['OFFENSE']=='HOMICIDE'], s=50, alpha=0.3, color='r', lw=0, label='Homicide')
plt.scatter(-77.03654,38.89722, s=70, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968, s=70, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right')
plt.show()

# ---<<< Correlation Matrix >>>---
def plot_corr(df, size = 15):
    global corr_df
    corr_df = df.select_dtypes(include=[float, int])
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax, shrink = .8)
    ax.grid(True)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 'vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

plot_corr(df)
corr_matrix = corr_df.corr()
corr_matrix.to_csv(r"C:\Users\Viraj\Desktop\Portfolio Projects\Python_CrimeAnalysis\DC_Crime_Correlation_Matrix.csv")