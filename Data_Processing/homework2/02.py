import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)

max_index = df['npop'].argmax()
if df.iloc[max_index,0]>df.iloc[max_index, 1]:
    print(df.iloc[max_index, 0])
else:
    print(df.iloc[max_index, 1])