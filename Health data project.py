#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# # Scenario 1
# ## Task-1

# In[2]:


# 1. load the data set

country=pd.read_csv('countryData.csv')
immun=pd.read_csv('immunitizationData.csv')


# In[3]:


# 2. exploring shape & data set

# shape
country.shape


# In[4]:


immun.shape


# In[5]:


# exploring data sets

country.head()


# In[6]:


immun.head()


# In[7]:


country.columns


# In[8]:


immun.columns


# In[9]:


# 3. Correct data issues if-any


# In[10]:


# Remove leading spaces from columns names
country.columns=country.columns.str.strip()
country.columns


# In[11]:


immun.columns=immun.columns.str.strip()
immun.columns


# In[12]:


# 4. Check missing values in all datasets.


# In[13]:


country.isnull().sum()


# In[14]:


immun.isnull().sum()


# In[15]:


# 5. Check duplicate values in all datasets.


# In[16]:


country.info()


# In[17]:


country.drop_duplicates()


# NO duplicate values in country table

# In[18]:


immun.info()


# In[19]:


immun.drop_duplicates()


# No duplicate values in immun column

# In[20]:


# 6. Analyze descriptive statistics.


# In[21]:


country.describe(include='all')


# In[22]:


immun.describe(include='all')


# # Task 2:

# In[23]:


# 1. Merge both the dataset with the unique key (create if not available). Also, remove duplicate columns.


# In[24]:


# creating unique key to join both data frames


# In[25]:


country['key']=country['Country']+'-'+country['Year'].astype(str)
country.head()


# In[26]:


immun['key']=immun['Country']+'-'+immun['Year'].astype(str)
immun.head()


# In[27]:


# Merging two data frames
df1=pd.merge(country, immun, on='key')


# In[28]:


df1.head()


# In[29]:


# checking column names, so we can remove duolicate and unnecessary columns
df1.columns


# In[30]:


# removing duplicate columns
df1.drop(['Country_y','Year_y','key'],axis=1,inplace=True)

# renaming columns COuntry and year as per origional data
df1.rename(columns={'Country_x':'Country','Year_x':'Year','Life expectancy':'Life_expectancy',
                   'Adult Mortality':'Adult_Mortality','Total expenditure':'Total_expenditure',
                   'thinness  1-19 years':'thinness_1_19_years', 'thinness 5-9 years':'thinness_5_9_years',
                   'Income composition of resources':'Income_composition_of_resources',
                   'infant deaths':'infant_deaths', 'Hepatitis B':'Hepatitis_B',
                   'under-five deaths':'under_five_deaths','HIV/AIDS':'HIV_AIDS'}, inplace=True)
df1.head()


# In[31]:


# exploring data frame
df1.describe(include='all')


# In[32]:


# as per above data it is clear that there are missing values in multiple columns , so we are calculating null values
df1.isnull().sum()


# In[33]:


df1.shape


# In[34]:


# remove duplicate entries
df1.drop_duplicates()


# In[35]:


#NO uplicate records found un data sets


# In[36]:


df1.info()


# # EDA
# 

# In[37]:


# 2. Find the continuous and categorical variables.


# In[38]:


categorical=df1.select_dtypes(include="O")
numerical=df1.select_dtypes(exclude="O")


# In[39]:


categorical.describe()


# In[40]:


numerical.describe()


# In[41]:


# 3. Impute missing value.


# In[42]:


df2=df1.fillna(df1.mean())
df2.info()


# # Scenario 2

# # task01

# In[43]:


# 1. univariate analysis


# In[44]:


#country
df2.Country.value_counts().tail(20)


# there are 10 countries who have appeared just once in this data sets.
# 
# 
# other 183 countries have frequency of 16 times

# In[45]:


# Year - Univariate analysis


# In[46]:


df2.Year.describe()


# In[47]:


sns.boxplot(x=df2['Year'])
plt.show()


# In[48]:


# Status - Univariate analysis


# In[49]:


df2.Status.value_counts()


# In[50]:


# finding total no. of developed and developing countries

df3 = df2[df2['Status'].isin(['Developing'])] 
df4 = df2[df2['Status'].isin(['Developed'])] 
print('Sum of Developing countries are= ', df3.Country.value_counts().count())
print('Sum of Developed countries are= ',df4.Country.value_counts().count())


# In[51]:


# Life expectancy univariate analysis


# In[52]:


df2.Life_expectancy.describe()


# In[53]:


df2.Life_expectancy.value_counts().head(10)


# In[54]:


sns.boxplot(x=df2['Life_expectancy'])
plt.show()


# In[55]:


# Adult_Mortality - Univariate analysis


# In[56]:


df2.Adult_Mortality.describe()


# In[57]:


sns.boxplot(x=df2['Adult_Mortality'])
plt.show()


# In[58]:


# Alcohol - Univariate analysis


# In[59]:


df2.Alcohol.describe()


# In[60]:


sns.boxplot(x=df2['Alcohol'])
plt.show()


# In[61]:


# Total_expenditure - Univariate analysis


# In[62]:


df2.Total_expenditure.describe()


# In[63]:


sns.boxplot(x=df2['Total_expenditure'])
plt.show()


# In[64]:


# GDP - Univariate analysis


# In[65]:


df2.GDP.describe()


# In[66]:


sns.boxplot(x=df2['GDP'])
plt.show()


# In[67]:


# Population - Univariate analysis


# In[68]:


df2.Population.describe()


# In[69]:


sns.boxplot(x=df2['Population'])
plt.show()


# In[70]:


# thinness_1-19_years - Univariate analysis


# In[71]:


df2.thinness_1_19_years.describe()


# In[72]:


sns.boxplot(x=df2['thinness_1_19_years'])
plt.show()


# In[73]:


# thinness_5-9_years - Univariate analysis


# In[74]:


df2.thinness_5_9_years.describe()


# In[75]:


sns.boxplot(x=df2['thinness_5_9_years'])
plt.show()


# In[76]:


# Income_composition_of_resources - Univariate analysis


# In[77]:


df2.Income_composition_of_resources.describe()


# In[78]:


sns.boxplot(x=df2['Income_composition_of_resources'])
plt.show()


# In[79]:


# Schooling - Univariate analysis


# In[80]:


df2.Schooling.describe()


# In[81]:


sns.boxplot(x=df2['Schooling'])
plt.show()


# In[82]:


# infant_deaths - Univariate analysis


# In[83]:


df2.infant_deaths.describe()


# In[84]:


sns.boxplot(x=df2['infant_deaths'])
plt.show()


# In[85]:


# Hepatitis_B - Univariate analysis


# In[86]:


df2.Hepatitis_B.describe()


# In[87]:


sns.boxplot(x=df2['Hepatitis_B'])
plt.show()


# In[88]:


# Measles - Univariate analysis


# In[89]:


df2.Measles.describe()


# In[90]:


sns.boxplot(x=df2['Measles'])
plt.show()


# In[91]:


# BMI - Univariate analysis


# In[92]:


df2.BMI.describe()


# In[93]:


sns.boxplot(x=df2['BMI'])
plt.show()


# In[94]:


# under_five_deaths - Univariate analysis


# In[95]:


df2.under_five_deaths.describe()


# In[96]:


sns.boxplot(x=df2['under_five_deaths'])
plt.show()


# In[97]:


# Polio - Univariate analysis


# In[98]:


df2.Polio.describe()


# In[99]:


sns.boxplot(x=df2['Polio'])
plt.show()


# In[100]:


# Diphtheria - Univariate analysis


# In[101]:


df2.Diphtheria.describe()


# In[102]:


sns.boxplot(x=df2['Diphtheria'])
plt.show()


# In[103]:


# HIV_AIDS - Univariate analysis


# In[104]:


df2.HIV_AIDS.describe()


# In[105]:


sns.boxplot(x=df2['HIV_AIDS'])
plt.show()


# In[106]:


df2.columns


# In[107]:


# 2. BIVARIATE ANALYSIS


# In[108]:


df_Life=df2.drop(['Year'], axis=1)

df_Life['Underweight']=np.where(df_Life['BMI']<18.5, 1, 0)
df_Life['Overweight']=np.where(df_Life['BMI']>24.9, 1, 0)
df_Life['Healthy']=np.where((df_Life['BMI']<24.9)& (df_Life['BMI']>18.5), 1, 0)
df_Life

# def weight(df):

#     if pd.isnull(df['height']):
#         return df['height']
#     elif (df['trigger1'] <= df['score'] < df['trigger2']) and (df['height'] < 8):
#         return df['height']*2
#     elif (df['trigger2'] <= df['score'] < df['trigger3']) and (df['height'] < 8):
#         return df['height']*3
#     elif (df['trigger3'] <= df['score']) and (df['height'] < 8):
#         return df['height']*4
#     elif (df['height'] > 8):
#         return np.nan


# In[109]:


# a. a.	Find the highly correlated variables.
corr=df_Life.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], ascending=False).reset_index()
corr


# In[110]:


# b.i. developed vs developing countries - GDP per capita of the country impact life expectancy


# In[111]:


sns.scatterplot(x= df2['GDP'], y= df2['Life_expectancy'] , hue= df2['Status'])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('GDP', fontsize=14, fontweight='bold')
plt.show()

o=df2.groupby(['Status'])['GDP','Life_expectancy'].mean().reset_index()
o.rename(columns={'Life_expectancy':'Avg_Life_expectancy','GDP':'Avg_GDP'},inplace=True)
o


# # insight:-
# After GDP range above 20000, the life expectancy is not below 70 years
# 

# In[112]:


# b.ii. developed vs developing countries - Income_composition_of_resources impact life expectancy


# In[113]:


sns.scatterplot(x= df2['Income_composition_of_resources'], y= df2['Life_expectancy'] , hue= df2['Status'])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Income_composition_of_resources', fontsize=14, fontweight='bold')
plt.show()

o=df2.groupby(['Status'])['Income_composition_of_resources','Life_expectancy'].mean().reset_index()
o.rename(columns={'Life_expectancy':'Avg_Life_expectancy','Income_composition_of_resources':'Income_composition_of_resources'},inplace=True)
o


# #### insight
# Life expectancy is proportional to Income_composition_of_resource

# In[114]:


# b.iii. developed vs developing countries - Schooling impact life expectancy


# In[115]:


sns.jointplot(x= df2['Schooling'], y= df2['Life_expectancy'] , hue= df2['Status'], kind = 'scatter')
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Schooling', fontsize=14, fontweight='bold')
plt.show()

sns.lineplot(x= df2['Schooling'], y= df2['Life_expectancy'] , hue= df2['Status'])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Schooling', fontsize=14, fontweight='bold')
plt.show()

o=df2.groupby(['Status'])['Schooling','Life_expectancy'].mean().reset_index()
o.rename(columns={'Life_expectancy':'Avg_Life_expectancy','Schooling':'Avg_Schooling'},inplace=True)
o


# In[116]:


# under_5 deaths vs schooling
sns.jointplot(x= df2['Schooling'], y= df2['under_five_deaths'] , hue= df2['Status'], kind = 'scatter')
plt.ylabel('under_five_deaths', fontsize= 14, fontweight='bold')
plt.xlabel('under_five_deaths', fontsize=14, fontweight='bold')
plt.show()
o=df2.groupby(['Status'])['Schooling','under_five_deaths'].mean().reset_index()
o.rename(columns={'Life_expectancy':'Avg_Life_expectancy','under_five_deaths':'Avg_under_five_deaths'},inplace=True)
o


# In[117]:


# thinness_1_19_years vs schooling
sns.jointplot(x= df2['Schooling'], y= df2['thinness_1_19_years'] , hue= df2['Status'], kind = 'scatter')
plt.ylabel('thinness_1_19_years', fontsize= 14, fontweight='bold')
plt.xlabel('thinness_1_19_years', fontsize=14, fontweight='bold')
plt.show()
o=df2.groupby(['Status'])['Schooling','thinness_1_19_years'].mean().reset_index()
o.rename(columns={'Life_expectancy':'Avg_Life_expectancy','thinness_1_19_years':'Avg_thinness_1_19_years'},inplace=True)
o


# In[118]:


# thinness_5_9_years vs schooling
sns.jointplot(x= df2['Schooling'], y= df2['thinness_5_9_years'] , hue= df2['Status'], kind = 'scatter')
plt.ylabel('thinness_5_9_years', fontsize= 14, fontweight='bold')
plt.xlabel('thinness_5_9_years', fontsize=14, fontweight='bold')
plt.show()
o=df2.groupby(['Status'])['Schooling','thinness_5_9_years'].mean().reset_index()
o.rename(columns={'Life_expectancy':'Avg_Life_expectancy','thinness_1_19_years':'Avg_thinness_5_9_years'},inplace=True)
o


# In[ ]:





# ## insight
# 
# > The shooling does increases life expectancy.
# In developing countries - Life expectancy s constantly increasing from schooling avg schooling i.e. 11 to 16(approximately).
# 
# >IN developed countries minimum schooling is above 10 and upto 20 and life expectancy increases with schooling
# 
# > Schooling reduces under five deaths , the rate of under five death is more in developing countries
# > Schooling reduces thinness in both 1-19 & 5-9 groups

# # 3. Multivariate analysis

# In[119]:


# a. Which diseases are correlated with life expectancy?


# In[120]:


ad=df2[['Status','Life_expectancy','thinness_1_19_years', 'thinness_5_9_years', 'infant_deaths',
       'Hepatitis_B', 'Measles', 'under_five_deaths', 'Polio',
       'Diphtheria', 'HIV_AIDS']]
ad


# In[121]:


ad.columns


# In[122]:


# aab=ad.plot(x='Life_expectancy', y=['thinness_1_19_years',
#        'thinness_5_9_years', 'infant_deaths', 'Hepatitis_B', 'Measles',
#        'under_five_deaths', 'Polio', 'Diphtheria', 'HIV_AIDS'],kind='bar')


# In[123]:


ad=df2[['Status','Life_expectancy','thinness_1_19_years', 'thinness_5_9_years', 'infant_deaths',
       'Hepatitis_B', 'Measles', 'under_five_deaths', 'Polio',
       'Diphtheria', 'HIV_AIDS']]
ad

cor=ad.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], ascending=True).reset_index()
cor.rename(columns={'index':'Name'},inplace=True)
cor


# In[124]:


af=cor.Name.values.tolist()
af.pop()
print('Diseases affecting the life expectancy are as below:- ')
for i in af:
    print(af.index(i)+1,' - ',i)


# In[125]:


# b. Does better immunization coverage improve life expectancy


# In[126]:


an=df2[['Status','Life_expectancy','GDP', 'Total_expenditure']]
an

cor=an.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], ascending=True).reset_index()
cor.rename(columns={'index':'Name'},inplace=True)
cor


# In[127]:



sns.jointplot(x= an['Total_expenditure'], y= an['Life_expectancy'] , hue= an['Status'], kind = 'scatter')
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Total_expenditure', fontsize=14, fontweight='bold')
plt.show()

cor=an.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], ascending=True).reset_index()
cor.rename(columns={'index':'Name'},inplace=True)
cor


# ### Insights:
# > The Income composition is directly proportional to life expectancy, but it is not afeecting in big way
# 
# > For low GDP countries, they will have to increase Total_expenditure

# In[128]:


# C. How do countries' economic conditions affect life expectancy


# In[129]:


sns.pairplot(df2[[ 'Status', 'Life_expectancy', 'Total_expenditure', 'GDP',]],hue='Status')
plt.show()


# In[130]:


sns.scatterplot(x= df2['GDP'], y= df2['Life_expectancy'] , hue= df2['Status'])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('GDP', fontsize=14, fontweight='bold')
plt.show()

o=df2.groupby(['Status'])['GDP','Life_expectancy'].mean().reset_index()
o.rename(columns={'Life_expectancy':'Avg_Life_expectancy','GDP':'Avg_GDP'},inplace=True)
o


# # insight
# 
# > if GDP increases the life expectancy also increases
# 
# > in countries who have GDP above 15000 life expectancy is expected to be above 70

# # Task 02

# ## a. Calculate the average life expectancy of all the years and find out Top and Bottom countries.

# In[131]:


# top 10 countries with high life expectancy
print("Top 10 Countries with high Life Expectancy")
df2.groupby("Country").agg({"Life_expectancy":"mean"
}).reset_index().sort_values(by=["Life_expectancy"], ascending = False).head(10)


# In[132]:


# top 10 countries with low life expectancy
print("Top 10 Countries with low Life Expectancy")
df2.groupby("Country").agg({"Life_expectancy":"mean"
}).reset_index().sort_values(by=["Life_expectancy"], ascending = True).head(10)


# ### Insights:-
# > country with highest life expectancy = Japan
# 
# > country with highest life expectancy = Sierra Leone

# ## b.	Rank countries based on their average life expectancy.

# In[133]:


a=df2.groupby(['Country'])['Life_expectancy'].mean().reset_index().sort_values(by=['Life_expectancy'],
                                                                              ascending=False)
a['Rank']=a['Life_expectancy'].rank(ascending=False).astype(int)
a


# # c. Compare a few country’s life expectancies.
# India, the United States of America, China, the Central African Republic, Bhutan 

# In[134]:


a=df2.Country.unique().tolist() # to check country name


# In[135]:


df_Country=df2[df2['Country'].isin(['India', 'United States of America', 'China', 'Central African Republic','Bhutan'])]
df_Country


# In[136]:


CO=df_Country.groupby(['Status'])['Life_expectancy'].mean().reset_index()
CO


# In[137]:


CO=df_Country.groupby(['Country','Status'])['Life_expectancy'].mean().reset_index()
CO


# In[138]:


sns.barplot(x= CO['Country'], y= CO['Life_expectancy'] , hue= CO['Status'])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Country', fontsize=14, fontweight='bold')
plt.show()


# # d. Compare life expectancy of Developed vs Developing country.

# In[139]:


Dev=df2.groupby(['Status'])['Life_expectancy'].mean().reset_index()
Dev


# In[140]:


a=sns.barplot(x= Dev['Status'], y= Dev['Life_expectancy'])
l=a.bar_label(a.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Status', fontsize=14, fontweight='bold')
plt.show()


# # 2. Let’s take few developing and developed countries and analyze them completely.

# In[141]:


df_Country=df2[df2['Country'].isin(['India', 'United States of America', 'China', 'Central African Republic','Bhutan'])]
df_Country


# In[142]:


cor=df_Country.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], ascending=True).reset_index()
cor.rename(columns={'index':'Name'},inplace=True)
cor


# In[143]:


# 3. Let’s look more
# a. Should a country having a lower life expectancy value(<65) increase its healthcare expenditure to improve its average lifespan?


# In[144]:


df_life=df_Country.groupby(['Country'])['Life_expectancy','Total_expenditure','BMI','Alcohol','Population'].mean().reset_index()
df_life


# In[145]:


df_life.sort_values(by=['Total_expenditure'], ascending=True,inplace=True)
df_life


# In[146]:


a=sns.barplot(x= df_life['Total_expenditure'], y= df_life['Life_expectancy'])
l=a.bar_label(a.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Total_expenditure', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# ### INSIGHT:
# > Countries with more higher expenditure has more life expectancy, so countries with life expectancy <65 should increase their Total expenditure

# In[147]:


# b. Does Life Expectancy have a positive or negative correlation with eating habits(BMI), lifestyle, exercise, smoking, drinking alcohol, etc?


# In[148]:


# correlation between data
cor=df_life.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], 
                                                    ascending=True).reset_index()
cor


# In[149]:


# BMI VS LIFE EXPECTANCY


# In[150]:


a=sns.barplot(x= df_life['BMI'], y= df_life['Life_expectancy'])
l=a.bar_label(a.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('BMI', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[151]:


f=df2.groupby(['Country'])['BMI','Life_expectancy','Alcohol'].mean().reset_index()
f


# In[152]:


# low BMI VS life expectancy
f.sort_values(by=['BMI'], ascending=True).reset_index()
l=f.head(10)
l


# In[153]:


a=sns.barplot(x=l['Life_expectancy'], y= l['BMI'])
l=a.bar_label(a.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('BMI', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[154]:


# High BMI VS life expectancy
m=f.sort_values(by=['BMI'], ascending=False).reset_index().head(10)
m


# In[155]:


b=sns.barplot(x=m.Life_expectancy, y= m.BMI)
l=b.bar_label(b.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('BMI', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[156]:


# Alcohol vs life expectancy


# In[157]:


# Low alcohol vs life expectancy
f.sort_values(by=['Alcohol'], ascending=True).reset_index()
l=f.head(10)
l


# In[158]:


a=sns.barplot(x=l.Life_expectancy, y=l.Alcohol)
l=a.bar_label(a.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Alcohol', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[159]:


# High Alcohol VS life expectancy
m=f.sort_values(by=['Alcohol'], ascending=False).reset_index().head(10)
m


# In[160]:


b=sns.barplot(x=m.Life_expectancy, y= m.Alcohol)
l=b.bar_label(b.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Alcohol', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# ### INSIGHT:
# > country with low BMI have low life expectancy
# 
# > Countries with high BMI has life expectancy above 65
# 
# > country with low Alcohol have low life expectancy
# 
# > Countries with high Alcohol has life expectancy above 65

# # c. Do densely populated countries tend to have a lower life expectancy?

# In[161]:


df_life.sort_values(by=['Population'], ascending=True,inplace=True)
df_life


# In[162]:


b=sns.barplot(x=df_life.Population, y= df_life.Life_expectancy)
l=b.bar_label(b.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Population', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[163]:


# low Population VS life expectancy
a=df2.groupby(['Country'])['Population','Life_expectancy'].mean().reset_index()
a
a.sort_values(by=['Population'], ascending=True).reset_index()
l=a.head(10)
l


# In[164]:


b=sns.barplot(x=l.Population, y=l.Life_expectancy)
l=b.bar_label(b.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Population', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[165]:


# High Population VS life expectancy

k=df2.groupby(['Country'])['Population','Life_expectancy'].mean().reset_index()
k
k.sort_values(by=['Population'], ascending=True).reset_index()
t=k.tail(10)
t


# In[166]:


c=sns.barplot(x=t.Population, y=t.Life_expectancy)
l=c.bar_label(c.containers[0])
plt.ylabel('Life Expectancy', fontsize= 14, fontweight='bold')
plt.xlabel('Population', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# ### INSIGHTS:
# 
# > Population does not have major impact on life expectancy

# # Colnclusion
# 1. predicting variables actually affecting life expectancy
# 

# In[167]:


cor=df2.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], ascending=True).reset_index()
cor.rename(columns={'index':'Name','Life_expectancy':'Impact on increasing life expectancy'},inplace=True)
cor


# 2. Should a country having a lower life expectancy value (<65) increase its healthcare expenditure to improve its average lifespan
# > The Income composition is directly proportional to life expectancy, but it is not effecting in big way
# 
# > For low GDP countries, they will have to increase Total_expenditure

# 3. How do Infant and Adult mortality rates affect life expectancy?
# 

# In[168]:


j=df2.groupby(['Country'])['Adult_Mortality','infant_deaths', 'Life_expectancy'].mean().reset_index()
j.sort_values(by=['Life_expectancy'], ascending=True, inplace=True)
j.corr()[['Life_expectancy']].sort_values(by=['Life_expectancy'], ascending=True)


# > Adult mortality and infant mortality are inversly proportional to Lifr expectany
# 
# > Adult mortality and infant mortality reduces life expectancy. Adult majority is major contributor
# 

# 4.	Does Life Expectancy have a positive or negative correlation with eating habits, lifestyle, exercise, smoking, drinking alcohol, etc?
# 
# > country with low BMI have low life expectancy
# 
# > Countries with high BMI has life expectancy above 65
# 
# > country with low Alcohol have low life expectancy
# 
# > Countries with high Alcohol has life expectancy above 65

# 5.	What is the impact of schooling on the lifespan of humans?
# 
# >The shooling does increases life expectancy. In developing countries - Life expectancy s constantly increasing from schooling avg schooling i.e. 11 to 16(approximately).
# 
# >IN developed countries minimum schooling is above 10 and upto 20 and life expectancy increases with schooling
# 
# >Schooling reduces under five deaths , the rate of under five death is more in developing countries Schooling reduces thinness in both 1-19 & 5-9 groups

# 6.	Does Life Expectancy have a positive or negative relationship with drinking alcohol?
# 
# > country with low Alcohol have low life expectancy
# 
# > Countries with high Alcohol has life expectancy above 65

# 7.	Do densely populated countries tend to have a lower life expectancy?
# > Population does not have major impact on life expectancy

# 8.	What is the impact of Immunization coverage on life Expectancy?
# 
# >Countries with more higher expenditure has more life expectancy, so countries with life expectancy <65 should increase their Total expenditure

# In[ ]:




