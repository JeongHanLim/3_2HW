import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

filename = "userXmovie.csv"
original_df = pd.read_csv(filename)

# For PCA, transpose data
df = original_df.transpose()
df.rename(columns = df.iloc[0, :], inplace = True)
df.columns = df.columns.astype('int64')
df = df.drop("Unnamed: 0", axis=0)

# Making Mean centered data
df = df-df.mean()
print(df)


#TODO: For Some reason, sum of df does not go to 0.

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDataFrame = pd.concat([principalDf, df], axis=1)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

plt.scatter(finalDataFrame.loc[:, "principal component 1"], finalDataFrame.loc[:, "principal component 2"])
plt.plot([x-16 for x in range(120)], [x for x in range(120)])
plt.plot([x-16 for x in range(120)], [-x for x in range(120)])

plt.show()