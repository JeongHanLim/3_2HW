#(g) Calculate the correlation between the difference in votes between Bush and Gore, and the votes obtained by Nader.
#
import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)
data = np.zeros(shape=(len(df),2))
df_corr = pd.DataFrame(data, columns = ['diff', 'nade'])
df_corr['diff'] = df.loc[:, 'bush']-df.loc[:, 'gore']
df_corr['nade'] = df.loc[:, 'nade']

print(df_corr.corr().iloc[0,1])