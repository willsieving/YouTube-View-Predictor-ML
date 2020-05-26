#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
# Np arrays won't print in scientific notation now


# In[36]:


data = pd.read_csv('youtube.csv')

data2 = data.sample(frac=1).reset_index(drop=True)
# frac = the fraction of values to put into the returned dataframe (100% of values, randomly suffled, with indexes from 1-end)
# Removing Values from Views
data3 = data2[data2.views != 0]

# Removing Values from Likes
data4 = data3[data2.likes != 0]

# Removing Values from Dislikes
data5 = data4[data2.dislikes != 0]

# Removing Values from Comments
data6 = data5[data2.comments != 0]


# In[37]:


features = np.log(data6.drop(['views'], axis=1))

log_views = np.log(data6['views'])
target = pd.DataFrame(log_views, columns=['views'])


# In[38]:


LIKES_IDX = 0
DISLIKES_IDX = 1
COMMENTS_IDX = 2

video_stats = features.mean().values.reshape(1, 3)


# In[39]:


regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

# Calculating the MSE and RMSE
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


# In[40]:


def get_log_estimate(likes=None,
                    dislikes=None,
                    comments=None,
                    high_confidence=True):
    # Configure video stats (use average or not)
    if likes:
        video_stats[0][LIKES_IDX] = likes
    else:
        pass
    
    if dislikes:
        video_stats[0][DISLIKES_IDX] = dislikes
    else:
        pass
    
    if comments:
        video_stats[0][COMMENTS_IDX] = comments
    else:
        pass
        
    # Make Prediction using Video Stats
    log_estimate = regr.predict(video_stats)[0][0]
    
    # Calculate the Range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return log_estimate, upper_bound, lower_bound, interval


# In[41]:


estimate = get_log_estimate(likes=14483, dislikes=379, comments=1839)
print(estimate)


# In[42]:


def get_views_estimate(vid_likes, vid_dislikes, vid_comments, large_range=True):
    """Estimate the views of a video on YouTube. Will work on any moderately popular video,
    especially videos on the Trending tab.
    Skews high, as such your prediction may be inside the range but near the lower bound with 95% coverage.
    
    Keyword arguments:
    like -- number of likes on the video.
    dislikes -- the number of dislikes on the video.
    comments -- the number of comments of the video.
    large_range -- True for a 95% prediction interval, False for a 68% interval (default is True)
    
    NOTE: Arguments cannot < 1
    
    """
    
    
    if vid_likes < 1 or vid_dislikes < 1 or vid_comments < 0:
        print('That is unrealistic. Try again.')
        return
    
    log_est, upper, lower, conf = get_log_estimate(likes = np.log(vid_likes), 
                                                   dislikes = np.log(vid_dislikes),
                                                   comments = np.log(vid_comments),
                                                   high_confidence = large_range)

    # Convert to normal views (non-log)
    views_est = np.e**log_est
    views_hi = np.e**upper
    views_low = np.e**lower

    # Round the views
    rounded_est = np.around(views_est, -3)
    rounded_hi = np.around(views_hi, -3)
    rounded_low = np.around(views_low, -3)


    print(f'The estimated view count is {rounded_est}')
    print(f'At {conf}% confidence the valuation range is')
    print(f'{rounded_low} views at the lower end to {rounded_hi} views at the high end.')


# In[43]:

# Actual: 12,592,175
get_views_estimate(vid_likes=1304199, vid_dislikes=12898, vid_comments=68051)


# In[45]:

# Actual: 1,236,437
get_views_estimate(vid_likes=82180, vid_dislikes=1841, vid_comments=3510)


# In[46]:

# Actual: 440,668
get_views_estimate(vid_likes=23630, vid_dislikes=860, vid_comments=2733)


