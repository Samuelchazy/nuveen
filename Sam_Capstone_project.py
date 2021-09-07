#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_style('dark')
import os
os.chdir('/Users/samuelchazy/ML_files_X/Capstone/Data')
#os.chdir('/Users/samuelchazy/EMA Aesthetics Dropbox/FAB info/ML_files_X/Capstone/Data')
os.getcwd()


# # <span style='color:Magenta'> Transaction Data File  </span>

# ## <span style='color:Blue'> 36 Rolling Months File</span>
# ### Numerical file

# In[2]:


path_D = 'Transaction_Data_20210128.xlsx'
Transaction_Rolling_Months = pd.read_excel(path_D,skiprows=1,sheet_name='36 Rolling Months')
Transaction_Rolling_Months = Transaction_Rolling_Months.drop('Unnamed: 0',axis=1)
Transaction_Rolling_Months.head().T


# ### Rename the Contact ID and Refresh Date columns

# In[3]:


Transaction_Rolling_Months = Transaction_Rolling_Months.rename(columns={'CONTACT_ID':'ID','refresh_date':'date'})


# In[4]:


Transaction_Rolling_Months.shape


# In[5]:


Transaction_Rolling_Months.info()


# In[6]:


Transaction_Rolling_Months.isna().sum()


# ### Run statistics on each and every numerical column

# In[7]:


def run_statistics(df):
    for col in df.columns:
        print('\n'+'='*50)        
        print(col)
        print('='*50)
        print(df[col].describe())
        print('')
        percent = ((df[col] == 0).sum() * 100) / len(df[col])
        print(f'=> => {percent} % of the rows are zeros <= <=')

run_statistics(Transaction_Rolling_Months.select_dtypes(include='number'))


# ### Drop unnecessary columns

# In[8]:


columns_to_drop = ['AUM','sales_12M','redemption_12M','redemption_rate']
Transaction_Rolling_Months = Transaction_Rolling_Months.drop(columns_to_drop,axis=1)


# ### Add Total Sales & Total Redemptions to the Dataframe

# In[9]:


Transaction_Rolling_Months = Transaction_Rolling_Months.sort_values(['ID','date'])

Transaction_Rolling_Months['total_annual_sales'] = Transaction_Rolling_Months.groupby('ID')['sales_curr'].rolling(12).sum().values
Transaction_Rolling_Months['total_annual_redemption'] = Transaction_Rolling_Months.groupby('ID')['redemption_curr'].rolling(12).sum().values


# In[10]:


Transaction_Rolling_Months


# ## <span style='color:Blue'> Rep Details File</span>
# ### advisors file
# 

# In[11]:


path_D = 'Transaction_Data_20210128.xlsx'
Transaction_Rep_Details = pd.read_excel(path_D,skiprows=1,sheet_name='Rep Details')
Transaction_Rep_Details = Transaction_Rep_Details.drop('Unnamed: 0',axis=1)
Transaction_Rep_Details.head()


# In[12]:


Transaction_Rep_Details.shape


# ### Delete unnecessary columns

# In[13]:


columns_to_drop = ['Office ID','Firm name']
Transaction_Rep_Details = Transaction_Rep_Details.drop(columns_to_drop,axis=1)
Transaction_Rep_Details = Transaction_Rep_Details.rename(columns={'Contact ID':'ID'})
Transaction_Rep_Details


# ### Check the value & unique counts for each and every categorical column

# In[14]:


def check_value_counts(df):
    for col in df.columns:
        print('\n'+'='*50)        
        print(col)
        print('')
        unique_counts = df[col].nunique()
        value_counts = df[col].value_counts().sum()
        
        print(f'{unique_counts} is the number of unique counts')
        print(f'{value_counts} is the total number of value counts')

check_value_counts(Transaction_Rep_Details.select_dtypes(include='object'))


# In[15]:


Transaction_Rep_Details['Channel'].value_counts().sort_values(ascending=False)


# In[16]:


Transaction_Rep_Details['Sub channel'].value_counts().sort_values(ascending=False)


# ### Drop firms that have less than 10% contribution & Encode categorical columns

# In[17]:


def encode_columns(df,col):
    advisor_df = df[['ID',col]].reset_index(drop=True)
    
    advisor_df['count'] = advisor_df[col].map(advisor_df[col].value_counts())
    advisor_df = advisor_df[advisor_df['count'] > int(advisor_df.shape[0]*0.01)]
    advisor_df = advisor_df.drop('count',axis=1)
    
    advisor_df = advisor_df.reset_index()
    advisor_df = advisor_df.set_index('ID')
    advisor_df = pd.get_dummies(advisor_df,drop_first=True)
    print(f'{col} shape: {advisor_df.shape}')
    return advisor_df

Firm_ID_advisors = encode_columns(Transaction_Rep_Details,'Firm ID')
Channel_advisors = encode_columns(Transaction_Rep_Details,'Channel')
Sub_Channel_advisors = encode_columns(Transaction_Rep_Details,'Sub channel')


# ### Merge the categorical columns

# In[18]:


advisor_df = pd.DataFrame(Transaction_Rep_Details['ID']).reset_index(drop=True)
advisor_df = advisor_df.merge(Firm_ID_advisors,on='ID',how='left')
advisor_df = advisor_df.merge(Sub_Channel_advisors,on='ID',how='left')
advisor_df = advisor_df.fillna(0)

advisor_df


# ## <span style='color:Blue'> Merge the 2 files Data Frames</span>
# 

# In[19]:


final_df = Transaction_Rolling_Months.merge(advisor_df,on='ID',how='left')
final_df = final_df.fillna(0)


# In[20]:


final_df


# ### Add year & month column

# In[21]:


filtered_df = final_df.copy()
filtered_df['year'] = filtered_df['date'].dt.year
filtered_df['month'] = filtered_df['date'].dt.month
filtered_df = filtered_df.drop('date',axis=1).reset_index(drop=True)
filtered_df = filtered_df[['year','month','ID']+list(filtered_df.iloc[:,1:-2])]


filtered_df


# In[22]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.lineplot(data=filtered_df,x='year',y='total_annual_sales',color='blue',alpha=0.3)
plt.xlabel('',fontsize=14)
plt.ylabel('Total Annual Sales in US Dollars',fontsize=14)
plt.title('Yearly evolution of sales',fontsize=21)
plt.tight_layout()

plt.savefig('evolution_of_sales.png');


# In[23]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.lineplot(data=filtered_df,x='year',y='total_annual_redemption',color='blue',alpha=0.3)
plt.xlabel('',fontsize=14)
plt.ylabel('Total Annual Redemption in US Dollars',fontsize=14)
plt.title('Yearly evolution of redemption',fontsize=21)
plt.tight_layout()

plt.savefig('evolution_of_redemption.png');


# ### Save the filtered excel file

# In[24]:


filtered_df.to_excel('capstone.xlsx',columns=filtered_df.columns)
print('done saving the file...')


# # <span style='color:Magenta'> Work with the filtered and cleaned excel file </span>

# ### Read the filtered excel file

# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_style('dark')
import os


# In[26]:


os.chdir('/Users/samuelchazy/ML_files_X/Capstone/Data')
#os.chdir('/Users/samuelchazy/EMA Aesthetics Dropbox/FAB info/ML_files_X/Capstone/Data')
os.getcwd()


# In[27]:


df = pd.read_excel('capstone.xlsx')
print('Done reading the file...')


# In[28]:


final_df = df.drop('Unnamed: 0',axis=1)
final_df


# ### Drop advisors with less than 3 years observations combined
# #### Filter the data based on year and month

# In[29]:


temp_df = final_df.copy()
temp_df['count'] = np.zeros(len(temp_df))

# Years are shifted by -1 month in this data

temp_df.loc[(temp_df['year']==2018) & (temp_df['month']==11),'count']=1
temp_df.loc[(temp_df['year']==2019) & (temp_df['month']==11),'count']=1
temp_df.loc[(temp_df['year']==2020) & (temp_df['month']==11),'count']=1
temp_df = temp_df[temp_df['count']==1]
temp_df = temp_df.drop(['month', 'count'],axis=1)


# #### Filter the data based on results having at least 3 years

# In[30]:


temp_df['count'] = temp_df['ID'].map(temp_df['ID'].value_counts())
temp_df = temp_df[temp_df['count'] >= 3]
temp_df = temp_df.drop('count',axis=1)
advisors_df = temp_df.copy()
advisors_df


# ### Split the Data into Holdout and Training set

# In[31]:


training_data = advisors_df[advisors_df['year'] < 2020]
hold_out_data_df = advisors_df[advisors_df['year'] == 2020]

print(training_data.shape)
print(hold_out_data_df.shape)


# ### recompose the training data into 2018 variables + 'predicted' 2019 values(y)

# In[32]:


# get the 'y' from the 2019 data
training_data_2019 = training_data[training_data['year']==2019][['ID','total_annual_sales']]
training_data_2019 = training_data_2019.rename(columns={'total_annual_sales':'y'})

# get the data from 2018
training_data_df = training_data[training_data['year']==2018]

# merge 2018 data with the 'y' from 2019
training_data_df = training_data_df.merge(training_data_2019,on='ID',how='inner')

# drop unnecessary columns
training_data_df = training_data_df.drop(['sales_curr','redemption_curr','year','index_y','index_x'],axis=1)

# set the index to 'ID'
training_data_df = training_data_df.set_index('ID')

training_data_df.head()


# # Explore the distributions of the training & test sets

# In[33]:


training_data_df['total_annual_sales'].describe()


# ### Explore the corrolation between the variables

# In[34]:


corr_data = training_data_df.copy()
corr_data = corr_data.iloc[:,0:31]
corr_data


# In[35]:


fig,ax = plt.subplots(figsize=(15,12),dpi=600)
ax = sns.heatmap(data=corr_data.corr(),annot=True,annot_kws={'fontsize':6},cmap='PuRd')
plt.title('Independant Variables Corrolation Matrix',fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('Features_corr.jpg');


# In[36]:


print('There is a very high corrolation betweeen aum_P_ALT & aum_AC_REAL_ESTAE, so we will drop one of them')

training_data_df = training_data_df.drop('aum_P_ALT',axis=1)


# ## Construct a descision tree model to subdivide the training data into 3 categories based on the total annual sales

# In[37]:


training_data_tree = training_data_df.loc[:,['total_annual_sales','y']]
tree_train = training_data_tree.copy()

X_tree = tree_train.drop('y',axis=1)
y_tree = tree_train['y']

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_tree,y_tree,test_size=0.25,random_state=20,shuffle=True)
model = DecisionTreeRegressor(criterion='mae',max_depth=3)
model.fit(X_train,y_train)

ax,fig = plt.subplots(figsize=(15,10),dpi=600)
ax = tree.plot_tree(model,filled=True,fontsize=12)

plt.tight_layout()
plt.savefig('tree.png');


# In[38]:


low_training_data_df = training_data_df[training_data_df['total_annual_sales'] <= 40000]
mid_training_data_df = training_data_df[(training_data_df['total_annual_sales'] > 40000)&(training_data_df['total_annual_sales'] < 200000)]
high_training_data_df = training_data_df[training_data_df['total_annual_sales'] >= 200000]

print(f'low_bound: {low_training_data_df.shape[0]} advisors')
print(f'mid_bound: {mid_training_data_df.shape[0]} advisors')
print(f'high_bound: {high_training_data_df.shape[0]} advisors')


# In[39]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(data=low_training_data_df['total_annual_sales'],bins=100,color='blue',alpha=0.3)


# In[40]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.boxenplot(data=low_training_data_df['total_annual_sales'])


# In[41]:


# low_training_data_df[low_training_data_df['total_annual_sales'] < 0]['total_annu al_sales'].dropna()


# In[42]:


below_zero = low_training_data_df[low_training_data_df['total_annual_sales'] < 0]['total_annual_sales'].dropna()
low_training_data_df = low_training_data_df.drop(below_zero.index,axis=0)


# In[43]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.boxenplot(data=low_training_data_df['total_annual_sales'])


# In[44]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(data=mid_training_data_df['total_annual_sales'],bins=100,color='blue',alpha=0.3)


# In[45]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.boxenplot(data=mid_training_data_df['total_annual_sales'])


# In[46]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(data=high_training_data_df['total_annual_sales'],bins=100,color='blue',alpha=0.3)


# In[47]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.boxenplot(data=high_training_data_df['total_annual_sales'])


# In[48]:


# high_training_data_df[high_training_data_df['total_annual_sales'] > 15000000]['total_annual_sales'].dropna()


# In[49]:


# outliers = high_training_data_df[high_training_data_df['total_annual_sales'] > 15000000]['total_annual_sales'].dropna()
# high_training_data_df = high_training_data_df.drop(outliers.index,axis=0)


# In[50]:


# ax,fig = plt.subplots(figsize=(15,5),dpi=600)
# ax = sns.boxenplot(data=high_training_data_df['total_annual_sales'])


# ## Standardize the 3 DataFrames

# In[51]:


low_training_data_norm = low_training_data_df.copy()
mid_training_data_norm = mid_training_data_df.copy()
high_training_data_norm = high_training_data_df.copy()

from sklearn.preprocessing import StandardScaler

def to_norm(data):
    scaler = StandardScaler().fit_transform(data.iloc[:,0:32])
    data.iloc[:,0:32] = scaler
    return data


# In[52]:


low_training_data_norm = to_norm(low_training_data_norm)
mid_training_data_norm = to_norm(mid_training_data_norm)
high_training_data_norm = to_norm(high_training_data_norm)


# ## Plot the variables of each category

# In[53]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.barplot(data=low_training_data_norm.iloc[:,:32],ci=None,color='Violet',saturation=0.5)
plt.xticks(fontsize=8,rotation=90)

plt.tight_layout()
plt.savefig('low_training_data_df.png');


# In[54]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=low_training_data_df['total_annual_sales'],
                  bins=9,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Total Annual Sales in US Dollars',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('low_annual_sales.png');


# In[55]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=low_training_data_df['total_annual_redemption'],
                  bins=6,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Total Annual Redemption in US Dollars',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('low_annual_redemption.png');


# In[56]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=low_training_data_df['no_of_assetclass_sold_12M_1'],
                  bins=4,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Number of Asset Classes sold in the last 12 months',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('low_assetclass.png');


# In[57]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.barplot(data=mid_training_data_norm.iloc[:,:32],ci=None,color='Violet',saturation=0.5)
plt.xticks(fontsize=8,rotation=90)

plt.tight_layout()
plt.savefig('mid_training_data_df.png');


# In[58]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=mid_training_data_df['total_annual_sales'],
                  bins=9,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Total Annual Sales in US Dollars',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('mid_annual_sales.png');


# In[59]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=mid_training_data_df['total_annual_redemption'],
                  bins=6,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Total Annual Redemption in US Dollars',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('mid_annual_redemption.png');


# In[60]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=mid_training_data_df['no_of_assetclass_sold_12M_1'],
                  bins=4,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Number of Asset Classes sold in the last 12 months',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('mid_assetclass.png');


# In[61]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.barplot(data=high_training_data_norm.iloc[:,:32],ci=None,color='Violet',saturation=0.5)
plt.xticks(fontsize=8,rotation=90)

plt.tight_layout()
plt.savefig('high_training_data_df.png');


# In[62]:


# high_training_data_norm[high_training_data_norm['total_annual_sales']<1000000]['total_annual_sales']


# In[63]:


decile1 = high_training_data_df[high_training_data_df['total_annual_sales']<4000000]
decile2 = high_training_data_df[high_training_data_df['total_annual_sales']>=4000000]

fig,[ax1,ax2] = plt.subplots(1,2,figsize=(15,5),dpi=600)

ax1 = sns.histplot(ax=ax1,x=decile1['total_annual_sales'],
                  bins=5,color='Violet',alpha=0.8,shrink=0.99)

ax1.set_xlabel(xlabel='Total Annual Sales in US Dollars > 200,000 & < 4,000,000',fontsize=14)
ax1.set_ylabel(ylabel='Number of Advisors',fontsize=14)
ax1.grid(axis='y',linewidth=2,color='lightgrey')

ax2 = sns.histplot(ax=ax2,x=decile2['total_annual_sales'],
                  bins=5,color='Violet',alpha=0.8,shrink=0.99)

ax2.set_xlabel(xlabel='Total Annual Sales in US Dollars >= 4,000,000',fontsize=14)
ax2.set_ylabel(ylabel='Number of Advisors',fontsize=14)
ax2.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('high_annual_sales.png');


# In[64]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=high_training_data_df['total_annual_redemption'],
                  bins=6,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Total Annual Redemption in US Dollars',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('high_annual_redemption.png');


# In[65]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.histplot(x=high_training_data_df['no_of_assetclass_sold_12M_1'],
                  bins=4,color='Violet',alpha=0.8,shrink=0.99)

plt.xlabel('Number of Asset Classes sold in the last 12 months',fontsize=14)
plt.ylabel('Number of Advisors',fontsize=14)
plt.grid(axis='y',linewidth=2,color='lightgrey')

plt.tight_layout()
plt.savefig('high_assetclass.png');


# ### Model Baseline Evaluation

# In[66]:


eval_model = pd.DataFrame(data={'loss':['low_mse','mid_mse','high_mse'],
                                'baseline':np.zeros(3),
                                'lasso':np.zeros(3),
                                'gbdt':np.zeros(3),
                                'rfr':np.zeros(3)})

low_y = low_training_data_norm['y']
mid_y = mid_training_data_norm['y']
high_y = high_training_data_norm['y']

low_yhat = low_training_data_norm['total_annual_sales']
mid_yhat = mid_training_data_norm['total_annual_sales']
high_yhat = high_training_data_norm['total_annual_sales']

mse_base_low = np.mean(np.square(low_y - low_yhat))
mse_base_mid = np.mean(np.square(mid_y - mid_yhat))
mse_base_high = np.mean(np.square(high_y - high_yhat))

eval_model.loc[0,'baseline'] = mse_base_low
eval_model.loc[1,'baseline'] = mse_base_mid
eval_model.loc[2,'baseline'] = mse_base_high

eval_model


# ### Apply L1(Lasso) regularization to the data and find the best alpha parameter

# In[67]:


from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error


# In[68]:


# Split the data

low_training_data_norm_X = low_training_data_norm.drop('y',axis=1)
low_training_data_norm_y = low_training_data_norm['y']

mid_training_data_norm_X = mid_training_data_norm.drop('y',axis=1)
mid_training_data_norm_y = mid_training_data_norm['y']

high_training_data_norm_X = high_training_data_norm.drop('y',axis=1)
high_training_data_norm_y = high_training_data_norm['y']


# ### Build the Lasso Models

# In[69]:


# Build the low_model

low_lasso = LassoCV(cv=3,max_iter=100000,fit_intercept=False,normalize=True,random_state=20)
low_lasso.fit(low_training_data_norm_X,low_training_data_norm_y)
print(list(low_lasso.coef_))

# Find the selected variables names
print('-'*80)
print('Columns with values > 0:')
print('-'*80)
cols = low_training_data_norm.columns.drop('y')
cols = list(cols[low_lasso.coef_ > 0])
print(cols)
print('-'*80)
low_training_data_norm_y_predicted = low_lasso.predict(low_training_data_norm_X)
mse_lasso_low = round(np.sqrt(mean_squared_error(low_training_data_norm_y,low_training_data_norm_y_predicted)),0)
print(f'MSE: {mse_lasso_low}')
print('-'*80)


# In[70]:


# Build the mid_model

mid_lasso = LassoCV(cv=3,max_iter=100000,fit_intercept=False,normalize=True,random_state=20)
mid_lasso.fit(mid_training_data_norm_X,mid_training_data_norm_y)
print(list(mid_lasso.coef_))

# Find the selected variables names
print('-'*80)
print('Columns with values > 0:')
print('-'*80)
cols = mid_training_data_norm.columns.drop('y')
cols = list(cols[mid_lasso.coef_ > 0])
print(cols)
print('-'*80)
mid_training_data_norm_y_predicted = mid_lasso.predict(mid_training_data_norm_X)
mse_lasso_mid = round(np.sqrt(mean_squared_error(mid_training_data_norm_y,mid_training_data_norm_y_predicted)),0)
print(f'MSE: {mse_lasso_mid}')
print('-'*80)


# In[71]:


# Build the high_model

high_lasso = LassoCV(cv=3,max_iter=100000,fit_intercept=False,normalize=True,random_state=20)
high_lasso.fit(high_training_data_norm_X,high_training_data_norm_y)
print(list(high_lasso.coef_))

# Find the selected variables names
print('-'*80)
print('Columns with values > 0:')
print('-'*80)
cols = high_training_data_norm.columns.drop('y')
cols = list(cols[high_lasso.coef_ > 0])
print(cols)
print('-'*80)
high_training_data_norm_y_predicted = high_lasso.predict(high_training_data_norm_X)
mse_lasso_high = round(np.sqrt(mean_squared_error(high_training_data_norm_y,high_training_data_norm_y_predicted)),0)
print(f'MSE: {mse_lasso_high}')
print('-'*80)


# ### Model Lasso Evaluation

# In[72]:


low_lasso_yhat = low_lasso.predict(low_training_data_norm_X)
mid_lasso_yhat = mid_lasso.predict(mid_training_data_norm_X)
high_lasso_yhat = high_lasso.predict(high_training_data_norm_X)

mse_lasso_low = np.mean(np.square(low_y - low_lasso_yhat))
mse_lasso_mid = np.mean(np.square(mid_y - mid_lasso_yhat))
mse_lasso_high = np.mean(np.square(high_y - high_lasso_yhat))

eval_model.loc[0,'lasso'] = mse_lasso_low
eval_model.loc[1,'lasso'] = mse_lasso_mid
eval_model.loc[2,'lasso'] = mse_lasso_high

eval_model


# ### Build a Gredient Boost Regressor model

# In[73]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


# In[74]:


def gbdt(params,X,y):
    min_params={'lrs':0,'n_ests':0,'subsamps':0,'maxdepths':0,'maxfeats':0}
    min_mse = np.Inf
    
    for lr in params['lrs']:
        for n_est in params['n_ests']:
            for subsamp in params['subsamps']:
                for maxdepth in params['maxdepths']:
                    for maxfeat in params['maxfeats']:
                        print(f'lr: {lr}')
                        print(f'n_est: {n_est}')
                        print(f'subsamp: {subsamp}')
                        print(f'maxdepth: {maxdepth}')
                        print(f'maxfeat: {maxfeat}')
                        print('-'*80)
                        
                        gbdt = GradientBoostingRegressor(learning_rate=lr,
                                                          n_estimators=n_est,
                                                          subsample   =subsamp,
                                                          max_depth   =maxdepth,
                                                          max_features=maxfeat)
                        scores=cross_val_score(gbdt,X,y,
                                scoring='neg_mean_squared_error',
                                cv=3,error_score='raise')
                        scores=scores*-1
                        print('mse'+str(np.mean(scores))+'\n')
                        if np.mean(scores)<min_mse:
                            min_mse=np.mean(scores)
                            min_params['lrs']     =lr
                            min_params['n_ests']  =n_est
                            min_params['subsamps']=subsamp
                            min_params['maxdepths']=maxdepth
                            min_params['maxfeats']=maxfeat
    return min_params


# In[75]:


params={'lrs':[0.0001],
       'n_ests':[2500,3500],
       'subsamps':[0.9,0.95],
       'maxdepths':[2,3],
       'maxfeats':[6,8]}


# In[76]:


low_params_gbdt = gbdt(params,low_training_data_norm_X,low_training_data_norm_y)
print('Done...')


# In[77]:


mid_params_gbdt = gbdt(params,mid_training_data_norm_X,mid_training_data_norm_y)
print('Done...')


# In[78]:


high_params_gbdt = gbdt(params,high_training_data_norm_X,high_training_data_norm_y)
print('Done...')


# In[79]:


# Save the best parameters

low_gbdt = GradientBoostingRegressor(learning_rate =low_params_gbdt['lrs'],
                                     n_estimators  =low_params_gbdt['n_ests'],
                                     subsample     =low_params_gbdt['subsamps'],
                                     max_depth     =low_params_gbdt['maxdepths'],
                                     max_features  =low_params_gbdt['maxfeats'])
low_gbdt.fit(low_training_data_norm_X,low_training_data_norm_y)

mid_gbdt = GradientBoostingRegressor(learning_rate =mid_params_gbdt['lrs'],
                                     n_estimators  =mid_params_gbdt['n_ests'],
                                     subsample     =mid_params_gbdt['subsamps'],
                                     max_depth     =mid_params_gbdt['maxdepths'],
                                     max_features  =mid_params_gbdt['maxfeats'])
mid_gbdt.fit(mid_training_data_norm_X,mid_training_data_norm_y)

high_gbdt = GradientBoostingRegressor(learning_rate=high_params_gbdt['lrs'],
                                     n_estimators  =high_params_gbdt['n_ests'],
                                     subsample     =high_params_gbdt['subsamps'],
                                     max_depth     =high_params_gbdt['maxdepths'],
                                     max_features  =high_params_gbdt['maxfeats'])
high_gbdt.fit(high_training_data_norm_X,high_training_data_norm_y)


# In[80]:


low_gbdt


# In[81]:


mid_gbdt


# In[82]:


high_gbdt


# ### Model GBDT Evaluation

# In[83]:


low_gbdt_yhat = low_gbdt.predict(low_training_data_norm_X)
mid_gbdt_yhat = mid_gbdt.predict(mid_training_data_norm_X)
high_gbdt_yhat = high_gbdt.predict(high_training_data_norm_X)

mse_gbdt_low = np.mean(np.square(low_y - low_gbdt_yhat))
mse_gbdt_mid = np.mean(np.square(mid_y - mid_gbdt_yhat))
mse_gbdt_high = np.mean(np.square(high_y - high_gbdt_yhat))

eval_model.loc[0,'gbdt'] = mse_gbdt_low
eval_model.loc[1,'gbdt'] = mse_gbdt_mid
eval_model.loc[2,'gbdt'] = mse_gbdt_high

eval_model


# ### Build a Random Forest model

# In[84]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# In[85]:


def rfr(params,X,y):
    min_params={'max_samples':0,'n_ests':0,'max_leaf_nodes':0,'maxdepths':0,'maxfeats':'0'}
    min_mse = np.Inf
    i=0
    
    for max_sample in params['max_samples']:
        for n_est in params['n_ests']:
            for max_leaf_node in params['max_leaf_nodes']:
                for maxdepth in params['maxdepths']:
                    for maxfeat in params['maxfeats']:
                        print(f'max_sample: {max_sample}')
                        print(f'n_est: {n_est}')
                        print(f'max_leaf_node: {max_leaf_node}')
                        print(f'maxdepth: {maxdepth}')
                        print(f'maxfeat: {maxfeat}')
                        print('-'*80)
                        
                        rfr = RandomForestRegressor(max_samples   =max_sample,
                                                    n_estimators  =n_est,
                                                    max_leaf_nodes=max_leaf_node,
                                                    max_depth     =maxdepth,
                                                    max_features  =maxfeat)
                        scores=cross_val_score(rfr,X,y,
                                scoring='neg_mean_squared_error',
                                cv=3,error_score='raise')
                        scores=scores*-1
                        i+=1
                        print(f'{i}: mse: {str(np.mean(scores))}\n')
                        if np.mean(scores)<min_mse:
                            min_mse=np.mean(scores)
                            min_params['max_samples']   =max_sample
                            min_params['n_ests']        =n_est
                            min_params['max_leaf_nodes']=max_leaf_node
                            min_params['maxdepths']     =maxdepth
                            min_params['maxfeats']      =maxfeat
    return min_params


# In[86]:


params={'max_samples':[0.5,0.8],
       'n_ests':[2500,3500],
       'max_leaf_nodes':[8,10],
       'maxdepths':[3],
       'maxfeats':['sqrt']}


# In[87]:


low_params_rfr = rfr(params,low_training_data_norm_X,low_training_data_norm_y)
print('Done...')


# In[88]:


mid_params_rfr = rfr(params,mid_training_data_norm_X,mid_training_data_norm_y)
print('Done...')


# In[89]:


high_params_rfr = rfr(params,high_training_data_norm_X,high_training_data_norm_y)
print('Done...')


# In[90]:


# Save the best parameters

low_rfr = RandomForestRegressor(     max_samples   =low_params_rfr['max_samples'],
                                     n_estimators  =low_params_rfr['n_ests'],
                                     max_leaf_nodes=low_params_rfr['max_leaf_nodes'],
                                     max_depth     =low_params_rfr['maxdepths'],
                                     max_features  =low_params_rfr['maxfeats'])
low_rfr.fit(low_training_data_norm_X,low_training_data_norm_y)

mid_rfr = RandomForestRegressor(     max_samples   =mid_params_rfr['max_samples'],
                                     n_estimators  =mid_params_rfr['n_ests'],
                                     max_leaf_nodes=mid_params_rfr['max_leaf_nodes'],
                                     max_depth     =mid_params_rfr['maxdepths'],
                                     max_features  =mid_params_rfr['maxfeats'])
mid_rfr.fit(mid_training_data_norm_X,mid_training_data_norm_y)

high_rfr = RandomForestRegressor(     max_samples  =high_params_rfr['max_samples'],
                                     n_estimators  =high_params_rfr['n_ests'],
                                     max_leaf_nodes=high_params_rfr['max_leaf_nodes'],
                                     max_depth     =high_params_rfr['maxdepths'],
                                     max_features  =high_params_rfr['maxfeats'])
high_rfr.fit(high_training_data_norm_X,high_training_data_norm_y)


# ### Model Random Forest Evaluation

# In[91]:


low_rfr_yhat = low_rfr.predict(low_training_data_norm_X)
mid_rfr_yhat = mid_rfr.predict(mid_training_data_norm_X)
high_rfr_yhat = high_rfr.predict(high_training_data_norm_X)

mse_rfr_low = np.mean(np.square(low_y - low_rfr_yhat))
mse_rfr_mid = np.mean(np.square(mid_y - mid_rfr_yhat))
mse_rfr_high = np.mean(np.square(high_y - high_rfr_yhat))

eval_model.loc[0,'rfr'] = mse_rfr_low
eval_model.loc[1,'rfr'] = mse_rfr_mid
eval_model.loc[2,'rfr'] = mse_rfr_high

eval_model


# In[92]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)

sns.kdeplot(data=eval_model,shade='fill')
plt.xlabel('Various ML models error(mean squared error) built on the 2018/2019 training data',fontsize=14)
plt.ylabel('Density curves distribution',fontsize=14)
plt.legend(['Baseline Model','Lasso Model','Gredient Boosting Decision Tree Model','Random Forest Model'])
plt.savefig('tables.png')
plt.show();


# # Build the validation Dataset

# In[93]:


# hold_out_data (2020)
hold_out_data = hold_out_data_df.copy()


# In[94]:


# Filter the holdout Dataset
hold_out_data = hold_out_data[['ID','total_annual_sales']]
hold_out_data = hold_out_data.rename(columns={'total_annual_sales':'y'})


# In[95]:


# get the 2019 data
data_2019_df = advisors_df[advisors_df['year']==2019]


# In[96]:


# Merge the 2019 dataset into the holdout dataset
hold_out_df = data_2019_df.merge(hold_out_data,how='inner',on='ID')
hold_out_df = hold_out_df.drop(['sales_curr','redemption_curr','year','index_x','index_y','aum_P_ALT'],axis=1)
hold_out_df = hold_out_df.set_index('ID')


# In[97]:


hold_out_df.shape


# In[98]:


# Divide the hold_out_data into 3 categories
low_hold_out_data = hold_out_df[hold_out_df['total_annual_sales'] <= 40000]
mid_hold_out_data = hold_out_df[(hold_out_df['total_annual_sales'] > 40000)&(hold_out_df['total_annual_sales'] < 200000)]
high_hold_out_data = hold_out_df[hold_out_df['total_annual_sales'] >= 200000]

print(f'low_bound: {low_hold_out_data.shape[0]} advisors')
print(f'mid_bound: {mid_hold_out_data.shape[0]} advisors')
print(f'high_bound: {high_hold_out_data.shape[0]} advisors')


# In[99]:


below_zero_val = low_hold_out_data[low_hold_out_data['total_annual_sales'] < 0]['total_annual_sales'].dropna()
low_hold_out_data = low_hold_out_data.drop(below_zero_val.index,axis=0)

# outliers_val = high_hold_out_data[high_hold_out_data['total_annual_sales'] > 15000000]['total_annual_sales'].dropna()
# high_hold_out_data = high_hold_out_data.drop(outliers_val.index,axis=0)


# In[100]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.boxenplot(data=low_hold_out_data['total_annual_sales'])


# In[101]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.boxenplot(data=mid_hold_out_data['total_annual_sales'])


# In[102]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
ax = sns.boxenplot(data=high_hold_out_data['total_annual_sales'])


# In[103]:


# Standardize the hold_out Dataset

low_hold_out_data_norm = low_hold_out_data.copy()
mid_hold_out_data_norm = mid_hold_out_data.copy()
high_hold_out_data_norm = high_hold_out_data.copy()

from sklearn.preprocessing import StandardScaler

def to_norm_holdout(data):
    scaler = StandardScaler().fit_transform(data.iloc[:,0:32])
    data.iloc[:,0:32] = scaler
    return data


# In[104]:


# Transform the Dataset

low_hold_out_data_norm = to_norm_holdout(low_hold_out_data_norm)
mid_hold_out_data_norm = to_norm_holdout(mid_hold_out_data_norm)
high_hold_out_data_norm = to_norm_holdout(high_hold_out_data_norm)


# In[105]:


# split the Dataset

low_hold_out_data_norm_X = low_hold_out_data_norm.drop('y',axis=1)
mid_hold_out_data_norm_X = mid_hold_out_data_norm.drop('y',axis=1)
high_hold_out_data_norm_X = high_hold_out_data_norm.drop('y',axis=1)


# ### Build the evaluation model

# In[116]:


holdout_eval_model = pd.DataFrame(data={'loss_2020':['low_mse','mid_mse','high_mse'],
                                'baseline_2020':np.zeros(3),
                                'gbdt_2020':np.zeros(3)})

low_y = low_hold_out_data_norm['y']
mid_y = mid_hold_out_data_norm['y']
high_y = high_hold_out_data_norm['y']

low_yhat = low_hold_out_data_norm['total_annual_sales']
mid_yhat = mid_hold_out_data_norm['total_annual_sales']
high_yhat = high_hold_out_data_norm['total_annual_sales']

mse_hold_base_low = np.mean(np.square(low_y - low_yhat))
mse_hold_base_mid = np.mean(np.square(mid_y - mid_yhat))
mse_hold_base_high = np.mean(np.square(high_y - high_yhat))

holdout_eval_model.loc[0,'baseline_2020'] = mse_hold_base_low
holdout_eval_model.loc[1,'baseline_2020'] = mse_hold_base_mid
holdout_eval_model.loc[2,'baseline_2020'] = mse_hold_base_high

mse_gbdt_low = np.mean(np.square(low_y - low_gbdt.predict(low_hold_out_data_norm_X)))
mse_gbdt_mid = np.mean(np.square(mid_y - mid_gbdt.predict(mid_hold_out_data_norm_X)))
mse_gbdt_high = np.mean(np.square(high_y - high_gbdt.predict(high_hold_out_data_norm_X)))

holdout_eval_model.loc[0,'gbdt_2020'] = mse_gbdt_low
holdout_eval_model.loc[1,'gbdt_2020'] = mse_gbdt_mid
holdout_eval_model.loc[2,'gbdt_2020'] = mse_gbdt_high

mse_rfr_low = np.mean(np.square(low_y - low_rfr.predict(low_hold_out_data_norm_X)))
mse_rfr_mid = np.mean(np.square(mid_y - mid_rfr.predict(mid_hold_out_data_norm_X)))
mse_rfr_high = np.mean(np.square(high_y - high_rfr.predict(high_hold_out_data_norm_X)))

holdout_eval_model.loc[0,'rfr_2020'] = mse_rfr_low
holdout_eval_model.loc[1,'rfr_2020'] = mse_rfr_mid
holdout_eval_model.loc[2,'rfr_2020'] = mse_rfr_high

holdout_eval_model


# In[118]:


all_eval_model = eval_model.join(holdout_eval_model)
all_eval_model = all_eval_model.drop(['loss_2020','lasso','gbdt','gbdt_2020'],axis=1)
all_eval_model


# In[119]:


ax,fig = plt.subplots(figsize=(15,5),dpi=600)
sns.kdeplot(data=all_eval_model,shade='fill')
plt.xlabel('Various ML models error(mean squared error)',fontsize=14)
plt.ylabel('Density curves distribution',fontsize=14)
plt.legend(['Baseline Model 2018/2019','Random Forest Regressor Model 2018/2019',
            'Baseline Model 2020','Random Forest Regressor Model 2020'])
plt.savefig('tables_final.png')
plt.show();


# # Construct Lift Charts

# In[122]:


# Add predicted y_hat to the Dataset

low_hold_out_data_norm['y_hat'] = low_rfr.predict(low_hold_out_data_norm_X)
mid_hold_out_data_norm['y_hat'] = mid_rfr.predict(mid_hold_out_data_norm_X)
high_hold_out_data_norm['y_hat'] = high_rfr.predict(high_hold_out_data_norm_X)


# In[123]:


# Function to create the Lift Charts
import math

def create_lift_chart(df):
    # add a decile column based on y_hat
    df['decile'] = pd.qcut(df['y_hat'],10,labels=False)
    
    # add n•_advisors column
    df['n•_advisors'] = 1
    
    # Capture the average
    avg_sales = np.mean(df['y'])
    
    # Group by decile to calculate the number of advisors & sales
    lift_df = df.groupby(['decile']).aggregate({'n•_advisors':np.sum,'y':np.mean})
    lift_df = lift_df.rename(columns={'y':'Sales_per_advisor'})
    lift_df['Sales_per_advisor'] = round(lift_df['Sales_per_advisor'],2)
    
    # Add lift
    lift_df['lift_percentage'] = np.round((lift_df['Sales_per_advisor'] - avg_sales) / avg_sales * 100,2)
    
    # sort the DataFrame and reset the index
    lift_df = lift_df.sort_values(by='Sales_per_advisor',ascending=False)
    lift_df = lift_df.reset_index()
    lift_df = lift_df.drop('decile',axis=1)
    lift_df.index = range(1,11)
    
    # Add cumulative number of advisors
    lift_df['n•_advisors_cumulative'] = 0
    lift_df.loc[1,'n•_advisors_cumulative'] = lift_df.loc[1,'n•_advisors']
    
    for i in range(len(lift_df['n•_advisors'])-1):
        lift_df.loc[i+2,'n•_advisors_cumulative'] = lift_df.loc[i+2,'n•_advisors'] + lift_df.loc[i+1,'n•_advisors_cumulative']
        
    # Add cumulative sales
    lift_df['Sales_per_advisor_cumulative'] = np.zeros(10)
    lift_df.loc[1,'Sales_per_advisor_cumulative'] = lift_df.loc[1,'Sales_per_advisor']
    
    for i in range(len(lift_df['n•_advisors'])-1):
        lift_df.loc[i+2,'Sales_per_advisor_cumulative'] = round(lift_df.loc[i+2,'Sales_per_advisor'].sum()/(i+2),2)
    
    # Add cumulative lift
    lift_df['lift_percentage_cumulative'] = np.zeros(10)
    lift_df.loc[1,'lift_percentage_cumulative'] = lift_df.loc[1,'lift_percentage']
    
    for i in range(len(lift_df['n•_advisors'])-1):
        lift_df.loc[i+2,'lift_percentage_cumulative'] = round(lift_df.loc[1:i+2,'lift_percentage'].sum()/(i+2),2)
    
    lift_df['n•_advisors'] = lift_df['n•_advisors'].apply(lambda x: str(f'{x:,}'))
    lift_df['Sales_per_advisor'] = lift_df['Sales_per_advisor'].apply(lambda x: str(f'{x:,}') + ' $')
    lift_df['lift_percentage'] = lift_df['lift_percentage'].apply(lambda x: str(x) + ' %')
    lift_df['n•_advisors_cumulative'] = lift_df['n•_advisors_cumulative'].apply(lambda x: str(f'{x:,}'))
    lift_df['Sales_per_advisor_cumulative'] = lift_df['Sales_per_advisor_cumulative'].apply(lambda x: str(f'{x:,}') + ' $')
    lift_df['lift_percentage_cumulative'] = lift_df['lift_percentage_cumulative'].apply(lambda x: str(x) + ' %')
        
    return lift_df
    


# ## Create the Lift Chart tables

# In[124]:


low_lift_df  = create_lift_chart(low_hold_out_data_norm)
mid_lift_df  = create_lift_chart(mid_hold_out_data_norm)
high_lift_df = create_lift_chart(high_hold_out_data_norm)


# In[125]:


#!pip install dataframe-image
import dataframe_image as dfi


# In[126]:


dfi.export(low_lift_df, 'low_lift_df_table.png')
low_lift_df


# In[127]:


dfi.export(mid_lift_df, 'mid_lift_df_table.png')
mid_lift_df


# In[128]:


dfi.export(high_lift_df, 'high_lift_df_table.png')
high_lift_df


# In[ ]:




