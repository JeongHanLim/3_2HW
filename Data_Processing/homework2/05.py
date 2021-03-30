import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)

co = 50

dict_var = {}
df_resize = df.loc[:, 'buch']

#print(df_resize)
buch_mean = df_resize.mean()

for i in range(len(df)):
    buch_value = df_resize.iloc[co-1]
    dict_var[i] = (np.square(df_resize.iloc[i]-buch_mean))
sorted_var = sorted(dict_var.items(), reverse = True, key = lambda x: x[1])
for index, values in enumerate(sorted_var):
    print("county_", values[0], "\t", values[1])