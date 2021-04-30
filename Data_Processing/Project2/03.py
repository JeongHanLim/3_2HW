import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

filename = "userXmovie.csv"
df = pd.read_csv(filename)
inertias=[]
def kmean(n_cluster, df):
    model = KMeans(n_clusters=n_cluster, algorithm="auto")
    model.fit(df)
    predict = pd.DataFrame(model.predict(df))
    predict.columns = ["predict"]
    df_labeled = pd.concat([df, predict], axis=1)
    inertias.append(model.inertia_)
    return df_labeled

labels = [2, 4, 8, 16, 32]
for i in labels:
    _ = kmean(i, df)

labels = [2, 4, 8, 16, 32]
plt.plot(labels, inertias, '-o')
plt.show()

df_labeled = kmean(8, df)
for i in range(8):
    temp_df = df_labeled.loc[df_labeled["predict"]==i]
    temp_df = temp_df.drop(["predict"], axis=1)
    temp_df = temp_df.drop(["Unnamed: 0"], axis=1)
    temp_df["average"] = temp_df.mean(1)
    temp_df = temp_df.sort_values("average", ascending=False)
    globals()["df{}".format(i)] = temp_df

print(df0.iloc[:3])
print(df1.iloc[:3])
print(df2.iloc[:3])
print(df3.iloc[:3])
print(df4.iloc[:3])
print(df5.iloc[:3])
print(df6.iloc[:3])
print(df7.iloc[:3])

