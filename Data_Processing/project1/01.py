import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import Preprocess
import time

filename = "input_data.csv"
df = pd.read_csv(filename)
DRG, idxDRG, DRG_num, idxDRG_num, ProvID = Preprocess.DRG().__getitem__()



def find_outlier(data,sigma=30): #Most data exists in 3sigma
    weirdrowmax, weirdrowmin = [], []
    maxsigma = np.mean(data)+sigma*np.var(data)/np.sqrt(len(data))
    minsigma = np.mean(data)-sigma*np.var(data)/np.sqrt(len(data))

    for i in range(len(data)):
        if data[i] > maxsigma:
            weirdrowmax.append((data[i]-np.mean(data))/np.var(data)*np.sqrt(len(data)))
        if data[i] < minsigma:
            weirdrowmin.append((data[i]-np.mean(data))/np.var(data)*np.sqrt(len(data)))
    return weirdrowmax,weirdrowmin


def del_dollar(df, type):
    return df.str.split(pat='$', expand=True).iloc[:, 1].astype(type)




if __name__ == "__main__":

    plt.plot(df.loc[:, " Total Discharges "])
    plt.title('Total Discharges on all Provider')
    plt.xlabel('ProviderID')
    plt.ylabel('Total Discharges')
    plt.show()
    df1 = df.groupby('DRG Definition')[" Total Discharges "].apply(list).reset_index(name = "Total Discharges")

    for i in range(len(df1)):
        tar = df1.loc[i, "Total Discharges"]
        plt.boxplot(tar)
        plt.xlabel('DRG '+str(DRG[i]))
        plt.ylabel('Total Discharges')
        plt.show()
        #When want to see more data other than DRG39, disable break.
        break

    #Just Looking at maximum peaking point.
    #Seems that only this data exceeds 3000,
    target_provider = 0
    idx_target = 0
    for i in range(len(df)):
        if df.loc[i, " Total Discharges "]>=3000:
            target_provider = df.iloc[i, 1]
            idx_target = i
    print(target_provider)
    print(df.iloc[idx_target, :])

    #Checking Evaluation of outlier
    index_470 = np.where(DRG_num==470)
    temp_list = []
    for i in range(index_470, len(index_470)):
        temp_list.append()















#df1["overnsigma"] = [[] for _ in range(100)]
#df1["undernsigma"]= [[] for _ in range(100)]
#print(df1["Total Discharges"])

#for i in range(len(df)):
#    a, b = find_outlier(df1.loc[i, "Total Discharges"])
#    print(i, a, b)
#    plt.plot(a)
#    plt.show()

#print(df1)




#plotting(df.loc[:, " Total Discharges "])
#df_ACC = del_dollar(df.loc[:, " Average Covered Charges "], float)
#plotting(df_ACC)
#df_ATP = del_dollar(df.loc[:, " Average Total Payments "], float)
#df_AMP = del_dollar(df.loc[:, "Average Medicare Payments"], float)
#plotting(df_ATP, df_AMP)
#plotting(df_ACC, df_AMP)

