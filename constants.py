# %%
import pandas as pd
data = pd.read_csv('bldgtype_country.csv')

labels = {}
for i in range(data.shape[0]):
    labels[data['Country'][i]] = list(data.columns[(data == 1).iloc[i]])
# %%
