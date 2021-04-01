import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)
#All possible candidates are 'bush', 'gore', 'brow', 'nade', 'harr', 'hage', 'buch', 'mcre', 'phil', 'moor'. You can find a person from the list.
df_resize = df.loc[:, ['bush', 'gore', 'brow', 'nade', 'harr', 'hage', 'buch', 'mcre', 'phil', 'moor']]

bush_won = 0
for len in range(len(df_resize)):
    if df_resize.loc[len,:].max() == df_resize.loc[len,'bush']:
        bush_won += 1
print(bush_won)


