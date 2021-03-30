#(h) Find the distance between the county that Bush won by the largest margin and the county that Gore won by the largest margin. (Just use basic Euclidean distance between the latitude (lat) and longitude (lon) values for the counties, no need to compute spherical distance.) (FYI: Eucliden distane is decribed in https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,metric%20as%20the%20Pythagorean%20metric.)

import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)

df_resize = df.loc[:, ['bush', 'gore', 'brow', 'nade', 'harr', 'hage', 'buch', 'mcre', 'phil', 'moor']]

df_resize['bush-2']=df_resize.loc[:, 'bush']-df_resize.apply(lambda row : row.nlargest(2).values[-1], axis=1)
df_resize['gore-2']=df_resize.loc[:, 'gore']-df_resize.apply(lambda row : row.nlargest(2).values[-1], axis=1)
print(df_resize['bush-2'].max())
print(df_resize['gore-2'].max())
