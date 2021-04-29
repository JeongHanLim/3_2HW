import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def transpose_dataframe(df):
    df = df.transpose()
    df=df.drop(["Unnamed: 0"])
    return df

# Making Mean centered data
def mean_centered(df):
    return df-df.mean()

def check_mean_centered(df):
    print(df.sum().sum())



if __name__=="__main__":
    filename = "userXmovie.csv"
    original_df = pd.read_csv(filename)
    df = transpose_dataframe(original_df)
    df = mean_centered(df)
    df=df.iloc[:, 0:2]
    #StandardScaler().fit(df) # or this.
    #print(df)
    principal = PCA(n_components=2)
    principal_components = principal.fit_transform(df)
    principaldf = pd.DataFrame(data=principal_components, index=df.index)
    principaldf.insert(0, "movieId", principaldf.index, True)
    principaldf["movieId"] = pd.to_numeric(principaldf["movieId"])

    filename2 = "movies.csv"
    df2 = pd.read_csv(filename2)
    df3 = pd.merge(principaldf, df2, left_on="movieId", right_on="movieId", how='outer')
    df3 = df3.sort_values("movieId")
    df3["genres"]=df3["genres"].str.split("|").str[0]
    genre = df3["genres"].tolist()
    unique_genre = df3["genres"].unique().tolist()
    genre_dict = {}
    df3["A"]=df3[0]
    df3["B"] = df3[1]
    print(df3)

    df3.plot.scatter(x="A", y="B")
    plt.show()
    print(df3[["A","B"]])
    # df3 = pd.concat([principaldf, df2["genres"]], axis=1)
    # print(df3)
