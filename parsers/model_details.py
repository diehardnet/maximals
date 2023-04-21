#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from profile_models import OUTPUT_DATABASE


# # Transformers models characteristics

# In[2]:


df_trans = pd.read_csv(OUTPUT_DATABASE)


# In[3]:


df_trans


# In[4]:


df_trans["para_by_depth"] = df_trans["layer_params"] * df_trans["depth"]


# In[5]:


gp_df_trans = df_trans.groupby(["net", "layer"]).mean()


# In[6]:


gp_df_trans = gp_df_trans[gp_df_trans['para_by_depth'] != 0]


# In[7]:


gp_df_trans.sort_values(by=["output_size"])


# In[8]:


df_trans["layer"].unique()


# In[21]:





# In[9]:


df_trans.groupby(["net", "layer"]).count().to_excel("/tmp/test.xlsx")


# In[22]:




