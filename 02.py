import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_DRG_index():
    DRGlist = df.iloc[:, 0].unique().tolist()
    DRGlist_num = np.zeros(len(DRGlist))

    for i in range(len(DRGlist)):
        DRGlist_num[i] = DRGlist[i].split(' ')[0].split("'")[0]
    DRGlist_num = DRGlist_num.astype(int)

    ProvID = df.iloc[:, 1].unique().tolist()
    return DRGlist_num,DRGlist, ProvID


def mknew_df(DRGlist, ProvID):
    new_column=["Prov.Id", "Prov.State"]

    for charges in DRGlist:
        name = str(charges)
        new_column.append(name)
    newdf = pd.DataFrame(columns=new_column)
    for i in range(len(ProvID)):
        newdf.loc[i] = [0 for x in range(102)]
        newdf.iloc[i, 0] = ProvID[i]
    return newdf


def insert_data(df, newdf, provid, idx, idx_char):
    for i in range(len(df)):
        id_idx = newdf.where(df.iloc[i, 0]) # Meaning [0,1,2,3./..] = [39, 54, ....]
        print(id_idx)
        #newdf.loc[newdf.where(df[i, 1]) , idx_char[id_idx]] = df.loc[i," Average Covered Charges "]

    return newdf


filename = "input_data.csv"
df = pd.read_csv(filename)

idx, idx_char, provid = get_DRG_index()
newdf = mknew_df(idx, provid)
newdf = insert_data(df, newdf, provid, idx, idx_char)
print(newdf)