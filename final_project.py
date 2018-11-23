
# coding: utf-8

# # Project: Investigate the 'No-show Patient' data sourced from Kaggle in order to answer fundamental questions about the data set.

# ## Introduction
# 

# For this report, a dataset from Kaggle titled, "No-show appointments" is explored and analyzed with the objective of determining if information in a patient profile can help predict predict their attendance to a scheduled appointment.
# 
# 100k scheduled medical appointments in Brazil is included in the dataset and focuses primarily on whether the patient showed up for their appointment or not.  Each row of data contains characteristics about the patient such as:
#  - 'ScheduledDay'
#      - What day the patient set up their appointment
#  - 'Neighbourhood'
#      - Location of the hospital
#  - 'Scholarship'
#      - Is the patient enrolled in Brasilian welfare program Bolsa Familia?
#  - 'No-show'
#      - If the patient didn't show up this entry is 'Yes'
# 
# The first question we wish to answer is which characteristic helps us predict the attendance of a patient the best.  A second question we would like to ask is what method helps us predict the solution with the highest accuracy and what are the difference in the results obtained by each method.
# 
# In order to determine the most influential characteristic on patient attendance we must clean the data so that an effective analysis can be performed.  Then we must select an effective analysis method to process our inputs into predictions.  Finally our analysis method should provide predictions that are accurate and reflect the reality of the scenario being considered to a certain degree of certainty.  This is to say we will validate the solution provided by two methods (the validation to compare effectiveness of method will be outside of the scope of this report):
# 
#  - Feature Selection by Importance
#  - SelectPercentile using Chi-squared
# 
# For both of these methods, no_show is provided as a label while all other characteristics of interest are formatted and regarded as features. This process is made possible by the train_test_split module from sklearn.model_selection. Feature_train and Label_train are the inputs used in both Chi2 and feature importance.
# 
# Feature selection by Importance is made possible by the inclusion of DecisionTreeClassifier from sklearn.tree module.  The attribute feature_importances returns the importance of the feature, such that importance is computed as the total reduction of criterion brought by that feature.
# 
# SelectPercentile using Chi-squared is provided by the feature_selection module in sklearn. Provided are the scores attributed to each feature.
# 
# A supplementary question we can answer is which day of the week represents the largest population of 'No-show's. This question is explored as a preliminary analysis before Feature Importance and Chi2 are performed.

# In[48]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import tree
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split


# ## Exploratory Data Analysis and Data Wrangling

# Create a dataframe of noshowappointents and convert the date column into a new column featuring the day of the week

# In[25]:


df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')


# In[26]:


print("Rows: ", df.shape[0])
print("Columns: ", df.shape[1])
df.head()


# The columns 'Gender' and 'No-show' are strings and will need to be converted into booleans in order to be used for machine learning.

# In[27]:


df.isnull().any()


# From above, it's clear there are no null cells and it seems this data set has been cleaned prior.

# In[28]:


#Referring to the prior reasoning, we need to convert our string cell entries in 'Gender' and 'No-shows' to booleans
df['Gender'] = df['Gender'].map({'M':1, 'F':0})
df['No-show'] = df['No-show'].map({'Yes':1, 'No':0})


# In[29]:


#Renaming columns to fix for typos and inconsistencies
df.rename(columns = {'AppointmentID': 'AppointmentId', 'Hipertension': 'Hypertension', 'Handcap': 'Handicap', }, inplace = True)
df.head()


# In[30]:


def classes_no_show():
    '''Perform a count of values of No-show to determine the distribution of classes'''
    no_show = 0
    for value in df['No-show']:
        if value == 1:
            no_show += 1
    print ("\n Out of", len(df['No-show']), "entries,", no_show, "didn't show, which in percentage of total is", round(float(no_show)/float(len(df['No-show'])), 4)*100, "%")

print ("Shown below is the value split in the 'No-show' column: 0 = patient showed up and 1 = patient did not show up")
print (df['No-show'].value_counts())
sns.countplot(x = 'No-show', data=df)

classes_no_show()


# There is an imbalance in classes of 'No-show'.  Quite a majority of patients showed up at their scheduled appointments. This information will provide a baseline for prediction of 'No-show' as is discussed later in this report in the section titled 'Null Accuracy'.  It is the baseline predictability of our model and is ~80% of patients show up.

# In[34]:


# Look into the range of 'Age' column to check for outliers
print ("Age range: ", sorted(df['Age'].unique()))


# The minimum age reported is -1 (this doesn't make much sense) and the maximum age reported is 115.  We cannot ascertain the true distribution of ages as this only represents all reported values without their frequency.

# In[32]:


# Distribution of 'Age' column
plt.figure();
age_hist = df['Age'].plot.hist(bins=10)
age_hist.set_xlabel("Age")
age_hist.set_ylabel("Patients")
age_hist.set_title('Distribution of Age')


# It's hard to ascertain the frequency of ages of -1 are in this dataset, however it seems the frequency of patients aged 90+ are quite few.  For this reason we will limit our data set to only include people aged 0-90 years.

# In[33]:


# Limit the patient ages to any age between 0-90 including 0 and 90.
df = df[(df.Age >= 0) & (df.Age <= 90)]

min_age = df['Age'].min()
max_age = df['Age'].max()
print("Age now is limited between and including: {} - {}.".format(min_age, max_age))


# In[35]:


# Let's look at the gender distribution now
print ("Distribution of patients, M or F: \n")
print ("Female: {} \nMale: {}".format(df['Gender'].value_counts()[0], df['Gender'].value_counts()[1]))

# Now let's plot the distribution of genders
gender_bar = df['Gender'].value_counts().plot.bar()
gender_bar.set_xticklabels(["Female", "Male"])
gender_bar.set_ylabel("Patients")
gender_bar.set_title('Distribution of Gender')


# ## Engineering the data

# We need to alter the formatting of our data so that all characteristics of interest can be processed.  To do this we must create new columns of data which contain information that can be used as inputs.

# In[58]:


# Let's transform the 'ScheduledDay' column and 'AppointmentDay' column into datetime objects and strip off the hours, minutes, and seconds.
dt_ScheduledDay = pd.to_datetime(df.ScheduledDay).dt.date
dt_AppointmentDay = pd.to_datetime(df.AppointmentDay).dt.date

# Let's create a delta feature to see difference between scheduled appointment days versus the realized appointment day.  We'll call it "delta_days"
df['Days_delta'] = (dt_AppointmentDay - dt_ScheduledDay).dt.days
df.head()


# In[ ]:


Below we will explore the distribution of weekdays where 'No-shows' took place. This will help supplement our understanding of why patients are missing their appointments.


# In[37]:


# Changing AppointmentDay to a datetime pandas object to create a new dayofweek engineered feature
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['No_show_weekday'] = df['AppointmentDay'].dt.dayofweek
sns.barplot(y='No-show', x='No_show_weekday', data=df)


# The highest proportion of no-shows coincide with when the appointment day happens to be on a Friday or Saturday.  It's apparent that on Sunday the clinic is closed (weekday = 6) as there is no data for that x value.  It looks from the graph, that the highest amount of no-shows occurs on a Friday or a Saturday.  This is emotionally what someone working a common 9-5 work week might expect and the data supports this assumption.

# In[41]:


# Now let's check out the distribution and data types of our data for modelling later on.

print ("PatientIds: ", len(df.PatientId.unique()))
print ("\nAppointmentIds: ", len(df.AppointmentId.unique()))
print ("\nNumber of appointment days: ", len(df.AppointmentDay.unique()))
print ("\nGender: ", df.Gender.unique())
print ("\nAge: ", sorted(df.Age.unique()), "\n Unique values: ", len(df.Age.unique()))
print ("\nSMS_received: ", df.SMS_received.unique())
print ("\nScholarship: ", df.Scholarship.unique())
print ("\nHypertension: ", df.Hypertension.unique())
print ("\nDiabetes: ", df.Diabetes.unique())
print ("\nAlcoholism: ", df.Alcoholism.unique())
print ("\nHandicap: ", df.Handicap.unique())
print ("\nNo-show: ", df["No-show"].unique())
print ("\nDays delta: ", sorted(df.Days_delta.unique()))
print ("\nBrazilian neighbourhoods: ", len(df.Neighbourhood.unique()))


# From above one can see that the proper format has been applied to all characteristics of interest such that analysis can be performed.  The spread of age is between 0 and 90 years of age.  A handicap value of '0' reflects no handicap where 1-4 are handicaps of four bins.  There are 81 discrete and unique neighbourhoods present in the data set.

# In[42]:


# days_delta contains various values which make no sense such as -1 and -6
# these look like mistakes 

days_hist = df['Days_delta'].plot.hist(bins=10)
days_hist.set_xlabel("Days delta")
days_hist.set_xticks(range(0, 180, 10))
days_hist.set_ylabel("Patients")
days_hist.set_title('Distribution of Days delta')


# In[43]:


# Remove days_delta values of negative value and greater than 90 as they represent extranneous ends of the data
df = df[(df.Days_delta >= 0) & (df.Days_delta <= 70)]
print ("Days delta: ", sorted(df.Days_delta.unique()))


# In[44]:


# Create variables to hold categorical features for one hot encoding
ctgry_features_for_encoding = ['Handicap', 'Neighbourhood']

# Creating variables for all numerical features to see
# which age and what days delta are the most important in predicting "No-show" outcome
num_features_for_encoding = ['Age', 'Days_delta']


# In[45]:


# One-Hot encoding on categorical columns to prep the dataset for machine learning modelling
encoded_df = pd.get_dummies(df, columns=ctgry_features_for_encoding)
print ("New encoded datafram has {} rows and {} features.".format(encoded_df.shape[0], encoded_df.shape[1]))

# Now we have to increase the maximum columns shown for this cell to 100
pd.set_option("max_columns", 100)

encoded_df.head()


# # Feature Selection

# In[46]:


# Dropping all varchar/date features including labels from df (a completed project I am referencing as a guide has dropped no_show_weekday without referencing the drop, therefore I will do the same)
features = encoded_df.drop(['No-show', 'No_show_weekday', 'PatientId', 'AppointmentId', 'ScheduledDay', 'AppointmentDay'], axis=1)

# labels will refer to "no-show" as they are the classes we are trying to predict
labels = encoded_df['No-show']
print(labels.head())
features.head()


# # Split features and label into train and test

# In[49]:


# Let's split the dataset into train and test for features and labels
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=60)


# # Check the Importance of Features

# In[50]:


# Fitting a tree
clf = tree.DecisionTreeClassifier(random_state=60)
clf.fit(features_train, labels_train)

# Feature Importances to check if new feature created has any importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print ("Rank the features: ")
for i in range(1, 11, 1):
    print (" {}. feature: {} ({})".format(i, features_train.columns[i], importances[indices[i]]))


# From the freature importances above you can see that the most important features are Age at 0.27, Scholarship at 0.05, Hypertension at 0.02, Diabetes at 0.01, and Alcoholism at 0.01.  The other features drop off in significance after this point.  One-hot encoding hasn't been employed on Age which would allow us to know which Age is the most useful in predicting a no-show.  Perhaps after this initial run, we can run One-hot encoding on Age as well.

# # SelectPercentile using Chi2

# Chi squared test is chosen as the function for statistical significance due to the nature of this dataset. This is because the features and labels contain no negative values and the majority of features are booleans.  Features that are independent of "No-show" are discarded and therefore irrelevant for classification and the scoring list is returned for each feature.

# In[51]:


# SelectPercentile to identify best features
selector = SelectPercentile(chi2)
selector.fit_transform(features_train, labels_train)

# Storing best features and their stores in separate pd.Series
scores = pd.Series(selector.scores_)
columns = pd.Series(features_train.columns)

# Concatenating both pd.Series into one df
selectbest = pd.concat([columns, scores], axis=1)
selectbest.rename(columns={0: 'features', 1: 'scores'}, inplace=True)


# In[52]:


# Visualizing best features using chi squared and their respective scores
selectbest_plot = selectbest.iloc[:8,:].plot.bar()
selectbest_plot.set_title('Feature Scoring')
selectbest_plot.set_xticklabels(selectbest['features'])
selectbest_plot.set_xlabel('Features')
selectbest_plot.set_ylabel('Score')


# The results presented above fit better with what one might predict as being important for a no-show.  Firstly and foremost, if a large amount of days have passed between the scheduling and actual appointment day, you might assume the patient could have likely forgotten the appointment, plans could have changed, an unplanned emergency took place, or various other complications.  Also, age is likely to play a role as teenagers and elderly people might be a bit less dependable based on maturity/responsibility and mental constitution/physical capability, respectively.  Finally, SMS_received would make sense, as if a reminder is not received the patient is likely to disregard the scheduled appointment as everyone nowadays pretty much lives and plans out of their phone or mobile device.

# Based on this reasoning, Chi-squared's SelectPercentile function is the best to use for feature selection.

# In[53]:


# Top features based on different methods are shown
best_chi_features = ['Days_delta', 'Age', 'SMS_received']
best_features_importances = ['Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism']
combined_best_features = ['Days_delta', 'Age', 'SMS_received', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism']


# # Null Accuracy

# In[54]:


# We will find the baseline accuracy possible of prediction if we took alone the value of "No-show"
labels_test.value_counts()


# In[56]:


# Distrubtion of no-show vs. showed up
print ("Proportion of patients that didn't show up", labels_test.mean())
print ("Proportion of patients that did show up", 1 -labels_test.mean())

# Null accuracy, accuracy achieved with a dumb model
print ("\nNull accuracy is", max(labels_test.mean(), 1 - labels_test.mean()))


# The minimum accuracy for our model is called the Null Accuracy.  This accuracy describes what the outcome would be considering our most frequently occurring classes only.

# The aim of our model is to provide a better accuracy than ~80%.
# 
# I will not run post processing as it is outside the scope of this project.

# #  Conclusion

# According to the method of feature selection on a decision tree, the top five most important features in determining a patient 'No-show' are: Age, Scholarship, Hypertension, Diabetes, and Alcoholism.  According to the SelectPrecentile with Chi-squared, the top features are: days_delta (engineered feature), Age, and SMS_received.  The Chi-squared method provided a more intuitive result and therefore will be relied upon for the conclusion of this project.
# 
# Feature Importance provides an importance for the feature of Age (importance = 0.28) which follows what one might expect as well as the results of the Chi2 analysis as well.  The second highest important feature provided is Scholarship (importance = 0.05) and reflects that to a minor degree a patients attendance in the Brasilian welfare program affects their attendance to a scheduled appointment.  This is perhaps because these patients are less likely to own a vehicle allowing them to travel to their appointment easily because of their conservative income. However, the other features reported are low in importance (importance < ~ .03) compared to age and might be rather ambiguous as none of them seem to definitively be strongly correlated to patient attendance.
# 
# The Chi-squared method provides a proportionally high score for days_delta (score = ~60000) which fits well with the assumption that a large amount of time between a scheduling day and the appointment day will increase the likelihood that a patient forgets their appointment and therefore doesn't show up.  Referring to the Distribution of days-delta graph earlier, you can see that a large amount of entries exist in the 30-70 days range, it might be desirable to reduce the amount of time between scheduling and appointment days to improve patient attendance. 
# 
# Relatively high scores are provided for Age (score = ~5000) and SMS_received (score = ~1000), as well, which suggests that dependability of patient attendance can be attributed to their age bracket (teenager vs. responsible adult vs. physically limited elderly individual).  It follows common reason to expect that the feature of SMS_received will impact largely the likelihood of a patient forgetting about their appointment and therefore not showing up. The other features scores drop below score of ~500 and are neglected as primary factors in predicting patient attendance.
# 
# Age is provided as a strong characteristic to predict 'No-show' by both methods and will be regarded as universally important in each method.  The feature 'days_delta' however is only deemed significant in the Chi2 method which suggests there's a reason for the difference.  It might be interesing in future analyses to determine why Feature Importance does not return this feature as important.
# 
# Additionally, it would be interesting to perform One-Hot Encoding on age to determine which age helps us determine the attendace of patients the best.  At ths point it's not considered and is outside the scope of this report.
# 
# Regarding our supplementary question, from our results it seems that when an appointment is scheduled for a Saturday, the patient is most likely to not show up.  It is hard to discern which other day of the week correlates with a high 'no-show' value.  It is therefore somewhat difficult to answer our question as all weekdays return a value somewhere between .18 and .23.
# 
# It appears the neighbourhood feature split into an encoded framework provides us little insight into the predictability of 'No-show'.  This might be because the variable has been spread so thin (n = 81) it loses its effect in the feature importance and Chi2 methods.  In future analyses, it might be interesting to determine which specific neighbourhoods are related to 'No-shows' greatest.  This phenemonon is possibly present in the feature importance and Chi2 scores of the handicap encoded framework for the same reason specified prior.
# 
# Additionally, a post processing to provide quantitated validation results comparing our two methods is desirable and was ultimately outside the scope of this report.  This section would provide us with a level of confidence we have in our models results.
# 
