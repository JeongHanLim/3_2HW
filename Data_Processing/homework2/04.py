#It asks to compute (the absolute difference / std).

import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)
co = 50

df_resize = df.loc[:, 'buch']
#print(df_resize)
buch_mean = df_resize.mean()
buch_std = df_resize.std()
print((df_resize.iloc[co-1]-buch_mean)/buch_std)
#print(np.square(df_resize.iloc[co-1]-buch_mean))
