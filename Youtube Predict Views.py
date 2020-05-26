#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


pd.set_option('display.max_rows', 50)

data = pd.read_csv('youtube.csv')
round(data.describe(), 1)


# ### Removing all the rows with value 0 to use a logarithmic transformation

# In[169]:


data2 = data.sample(frac=.5).reset_index(drop=True)
# frac = the fraction of values to put into the returned dataframe (100% of values, randomly suffled, with indexes from 1-end)
# Removing Values from Views
data3 = data2[data2.views != 0]
data3 = data3[data2.views != 1]
data3 = data3[data2.views != 2]
data3 = data3[data2.views != 3]
data3 = data3[data2.views != 4]
data3 = data3[data2.views != 5]
data3 = data3[data2.views != 6]
data3 = data3[data2.views != 7]
data3 = data3[data2.views != 8]
data3 = data3[data2.views != 9]
data3 = data3[data2.views != 10]

# Removing Values from Likes
data4 = data3[data2.likes != 0]
data4 = data4[data2.likes != 1]
data4 = data4[data2.likes != 2]
data4 = data4[data2.likes != 3]
data4 = data4[data2.likes != 4]
data4 = data4[data2.likes != 5]
data4 = data4[data2.likes != 6]
data4 = data4[data2.likes != 7]
data4 = data4[data2.likes != 8]
data4 = data4[data2.likes != 9]
data4 = data4[data2.likes != 10]

# Removing Values from Dislikes
data5 = data4[data2.dislikes != 0]
data5 = data5[data2.dislikes != 1]
data5 = data5[data2.dislikes != 2]
data5 = data5[data2.dislikes != 3]
data5 = data5[data2.dislikes != 4]
data5 = data5[data2.dislikes != 5]
data5 = data5[data2.dislikes != 6]
data5 = data5[data2.dislikes != 7]
data5 = data5[data2.dislikes != 8]
data5 = data5[data2.dislikes != 9]
data5 = data5[data2.dislikes != 10]

# Removing Values from Comments
data6 = data5[data2.comments != 0]
data6 = data6[data2.comments != 1]
data6 = data6[data2.comments != 2]
data6 = data6[data2.comments != 3]
data6 = data6[data2.comments != 4]
data6 = data6[data2.comments != 5]
data6 = data6[data2.comments != 6]
data6 = data6[data2.comments != 7]
data6 = data6[data2.comments != 8]
data6 = data6[data2.comments != 9]
data6 = data6[data2.comments != 10]

print(round(data6.describe(), 1))


# In[170]:


# In Seaborn
plt.figure(figsize=(10, 6))
ax = sns.distplot(data6['views'], hist=True, kde=True)
ax.set(xlabel='Views', ylabel='Number of Occurences', title='Views before Transformation')
plt.show()
plt.figure(figsize=(10, 6))
ax2 = sns.distplot(np.log(data6['views']), hist=True, kde=True)
ax2.set(xlabel='Log Views', ylabel='Number of Occurences', title='Views after Log Transformation')
plt.show()


# ## Testing & Training Datasets

# In[171]:


views = data6['views']
features = data6.drop('views', axis=1)

X_train, X_test, y_train, y_test = train_test_split(np.log(features), views, 
                                                    test_size=0.2, random_state=10)

# % of training set
print(len(X_train)/len(features))
print(X_test.shape[0]/features.shape[0])


# ## Multivariable Regression

# In[172]:


regr = LinearRegression()
regr.fit(X_train, y_train)

print('Training Data R-Squared:', regr.score(X_train, y_train))
print('Testing Data R-Squared:', regr.score(X_test, y_test))

print('Intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['Thetas'])


# ### Skew (get as close to 0 as possible) (Normal Distribution Skew = 0)

# In[173]:


y_log = np.log(data6['views'])
# Transform data to reduce influence of outliers
print(f'Non-Transfromed Skew: {data6["views"].skew()} \n vs. \nLog Transformed Skew: {y_log.skew()}')


# In[174]:


sns.distplot(y_log)
plt.show()


# ### How Transformation Effects Graph Curve (straightens it)

# In[175]:


# Scatter Plot without Transformation
sns.lmplot(x='comments', y='views', data=data6, height=7,
           scatter_kws={'alpha':0.6}, line_kws={'color':'darkred'})
plt.show()


# In[177]:


# Scatter Plot after Transformation
transformed_data = np.log(features)
transformed_data['log_views'] = y_log

sns.lmplot(x='comments', y='log_views', data=transformed_data, height=7,
           scatter_kws={'alpha':0.2}, line_kws={'color':'cyan'})
plt.show()


# ### Regression Using LOG Values

# In[181]:


views = np.log(data6['views']) # USE Log Prices
features = np.log(data6.drop('views', axis=1))

X_train, X_test, y_train, y_test = train_test_split(features, views, 
                                                    test_size=0.2, random_state=10)
regr = LinearRegression()
regr.fit(X_train, y_train)

print('Training Data R-Squared:', regr.score(X_train, y_train))
# -Training data is 75% accurate
print('Testing Data R-Squared:', regr.score(X_test, y_test))
# -Testing data is 67% accurate

print('Intercept', regr.intercept_)
print(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))


# In[182]:


X_incl_const = sm.add_constant(X_train)

print(X_incl_const)

model = sm.OLS(y_train, X_incl_const)
results = model.fit()

pd.DataFrame({'coef': results.params, 'p-value': results.pvalues})


# In[183]:


variance_inflation_factor(exog=X_incl_const.values, exog_idx=1)


# In[185]:


# Using List Comprehensions to create list of VIF numbers
vif = [variance_inflation_factor(exog=X_incl_const.values, exog_idx=i) for i in range(len(X_incl_const.columns))]

# Creating a Dataframe with Feature name and their VIF numbers
pd.DataFrame({'coef_ name': X_incl_const.columns,
             'vif': np.around(vif, 2)})


# ### Use BIC to consider dropping a feature

# In[186]:


# Log transformed Model
X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_incl_const)
results = model.fit()

org_coef = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 6)})

print('BIC Value:', results.bic)
print('R-Squared Value', results.rsquared)


# In[190]:


# Log Transformed Model w/ Comments dropped
X_incl_const = sm.add_constant(X_train)
X_incl_const = X_incl_const.drop(['comments'], axis=1)

model = sm.OLS(y_train, X_incl_const)
results = model.fit()

coef_minus_indus = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 6)})

print('BIC Value:', results.bic)
print('R-Squared Value', results.rsquared)


# In[191]:


# No Feature need to be dropped (we don't have a lot anyway)
# BIC always goes up, never down, when a feature is dropped here
# BIC needs to go down to drop a feature


# ### Graphing Residuals

# In[195]:


viewss = np.log(data6['views']) # USE Log Prices
features = np.log(data6.drop(['views'], axis=1))

X_train, X_test, y_train, y_test = train_test_split(features, views, 
                                                    test_size=0.2, random_state=10)

# Using Statsmodel
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

# Graph of Actual vs. Predicted Prices
corr = round(y_train.corr(results.fittedvalues), 2)
plt.figure(figsize=(8, 6))
plt.scatter(x=y_train, y=results.fittedvalues, c='navy', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
# Cyan line is perfect predictions

plt.xlabel('Actual log views $y _i$', fontsize=14)
plt.ylabel('Predicted log views $\hat y _i$', fontsize=14)
plt.title(f'Actual vs Predicted log views: $y _i$ vs $\hat y _i$ (Corr {corr})', fontsize=17)

plt.figure(figsize=(8, 6))
plt.scatter(x=np.e**y_train, y=np.e**results.fittedvalues, c='green', alpha=0.6)
plt.plot(np.e**y_train, np.e**y_train, color='cyan')
# Cyan line is perfect predictions

plt.xlabel('Actual views $y _i$', fontsize=14)
plt.ylabel('Predicted prices $\hat y _i$', fontsize=14)
plt.title(f'Actual vs Predicted views: $y _i$ vs $\hat y _i$ (Corr {corr})', fontsize=17)

plt.show()

# Residuals vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(x=results.fittedvalues, y=results.resid, c='navy', alpha=0.6)

plt.xlabel('Predicted log views $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Fitted Values', fontsize=17)

plt.show()

# Mean Squared Error & R-Squared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)


# In[196]:


# Distribution of Residuals (log views) - checking for normality
# Skew and Mean should be ~0
resid_mean = round(results.resid.mean(), 3) 
resid_skew = round(results.resid.skew(), 3)

plt.figure(figsize=(10,6))
sns.distplot(results.resid, color='navy')
plt.title(f'Log price model: Residuals (Skew = {resid_skew}, Mean = {resid_mean})')
plt.show()


# ### Calculating Standard Deviation and Range

# In[199]:


print('1 S.D. in log views is', np.sqrt(reduced_log_mse))
print('2 S.D. in log views is', 2*np.sqrt(reduced_log_mse))
rmse = np.sqrt(reduced_log_mse)

lower_bound = (np.log(100) - 2*rmse)
print('The lower bound in log views for a 95% prediction interval is:', lower_bound)
print('The lower bound in normal views is:', np.e**(lower_bound))

upper_bound = (np.log(100) + 2*rmse)
print('The upper bound in log views for a 95% prediction interval is:', upper_bound)
print('The upper bound in normal views is:', np.e**(upper_bound))


# In[ ]:




