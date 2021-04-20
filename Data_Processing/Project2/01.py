import pandas as pd
import numpy as np

filename = "ml-latest-small//"+"ratings.csv"
df = pd.read_csv(filename)

#Checked
userid_unique = df["userId"].unique().tolist()
movieid_unique = df["movieId"].unique().tolist()
print(len(userid_unique))
print(len(movieid_unique))

new_df = pd.DataFrame(0, columns = userid_unique, index = movieid_unique)
for i in range(len(df)):
    new_df.loc[df.loc[i, "movieId"],df.loc[i, "userId"]]\
        = df.loc[i, "rating"]
print(new_df)
new_df.to_csv("userXmovie.csv")
#TODO: Check nonzero data, which should be equal to 100836