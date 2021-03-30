import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)

df_resize = df.loc[:, 'buch']
#print(df_resize)
print(df_resize.mean())