import pandas as pd
import numpy as np

RATINGS_FILE_NAME = "ratings.csv"
MOVIES_FILE_NAME = "movies.csv"
ratings_df = pd.read_csv(RATINGS_FILE_NAME)
movies_df = pd.read_csv(MOVIES_FILE_NAME)

#Checked
userid_unique = ratings_df["userId"].unique().tolist()
movieid_unique = movies_df["movieId"].unique().tolist()
print(len(userid_unique))
print(len(movieid_unique))

new_df = pd.DataFrame(0, columns=movieid_unique, index=userid_unique)
for i in range(len(ratings_df)):
    new_df.loc[ratings_df.loc[i, "userId"],ratings_df.loc[i, "movieId"]] = ratings_df.loc[i, "rating"]
print(new_df)
new_df.to_csv("userXmovie.csv")
#TODO: Check nonzero data, which should be equal to 100836