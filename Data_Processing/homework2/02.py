import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)

max_index = df['npop'].argmax()


if (df.loc[max_index,'bush']<df.loc[max_index, 'gore']):
    print("Gore")
else:
    print("Bush")