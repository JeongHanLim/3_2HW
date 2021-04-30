import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RATINGS_FILE_NAME = "ratings.csv"
MOVIES_FILE_NAME = "movies.csv"
Q1_PROCESSED = "userXmovie.csv"
ratings_df = pd.read_csv(RATINGS_FILE_NAME)
movies_df = pd.read_csv(MOVIES_FILE_NAME)
user_rating_df = pd.read_csv(Q1_PROCESSED)

def transpose_dataframe(df):
    df = df.transpose()
    print(df)
    df = df.drop(["Unnamed: 0"])
    return df

# Making Mean centered data
def mean_centered(df):
    return df-df.mean()

def check_mean_centered(df):
    print(df.sum().sum())

def make_PCA(df):
    principal = PCA(n_components=2)
    principal_components = principal.fit_transform(df)
    principaldf = pd.DataFrame(data=principal_components, columns=["component1", "component2"])
    principaldf.insert(0, "movieId", df.index, True)
    principaldf["movieId"] = pd.to_numeric(principaldf["movieId"])

    return principaldf

def merge_movie(df1, df2):
    target_df = pd.merge(df1, df2, left_on="movieId", right_on="movieId", how='inner')
    target_df = target_df.sort_values("movieId")

    return target_df

def postprocess(df):
    df["genres"] = df["genres"].str.split("|").str[0]
    genre = df["genres"].tolist()
    unique_genre = df["genres"].unique().tolist()
    genre_dict = {string: i for i, string in enumerate(unique_genre)}
    genre_id = []
    for i in range(len(genre)):
        genre_id.append(genre_dict[genre[i]])

    return df, genre_id

def visualize(merged_df, genre_id):
    plt.scatter(merged_df["component1"].tolist(), merged_df["component2"].tolist(), c=genre_id, cmap=plt.cm.rainbow)
    plt.show()


df              = transpose_dataframe(user_rating_df)
df              = mean_centered(df)
principaldf     = make_PCA(df)
merged_df       = merge_movie(principaldf, movies_df)
merged_df, genre_id = postprocess(merged_df)
visualize(merged_df, genre_id)
