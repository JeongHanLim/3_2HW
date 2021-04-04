import pandas as pd
import numpy as np
#default
filename = 'bush-gore-results-fl_demo.csv'
df = pd.read_csv(filename)

df_feature = df.loc[:, ['whit','blac','hisp']]
candidate = ['bush', 'gore']
df_cand = df.loc[:, candidate]
cand_feat = [[0,0,0], [0,0,0]]

feature = ['whit', 'blac', 'hisp']
for i in range(len(df_cand)):
    for j in range(len(feature)):
       cand_feat[0][j] += df_feature.iloc[i, j] / 100 * df_cand.iloc[i, 0]
       cand_feat[1][j] += df_feature.iloc[i, j] / 100 * df_cand.iloc[i, 1]

cand_feat = np.asarray(cand_feat)
for i in range(len(feature)):
    print(feature[i], " ", candidate[cand_feat.argmax(axis=0)[i]])