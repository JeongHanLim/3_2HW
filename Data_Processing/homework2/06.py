import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)

df_resize = df.loc[:, ['whit','blac','hisp']]
df_idxmax=(df_resize.idxmax(axis=1))

str_list = ["white", "black", "hispanic"]
for idx ,values in enumerate(df_idxmax):
    values=values.replace("whit", str_list[0])
    values=values.replace("blac", str_list[1])
    values=values.replace("hisp", str_list[2])
    print(idx, values)