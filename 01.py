import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plotting(target, target1=None):
    plt.plot(target)
    if target1 is not None:
        plt.plot(target1)
    plt.show()

def del_dollar(df, type):
    return df.str.split(pat='$', expand=True).iloc[:, 1].astype(type)


filename = "input_data.csv"
df = pd.read_csv(filename)


plotting(df.loc[:, " Total Discharges "])
df_ACC = del_dollar(df.loc[:, " Average Covered Charges "], float)
plotting(df_ACC)
df_ATP = del_dollar(df.loc[:, " Average Total Payments "], float)
df_AMP = del_dollar(df.loc[:, "Average Medicare Payments"], float)
plotting(df_ATP, df_AMP)
plotting(df_ACC, df_AMP)

"""
plt.plot(df.loc[:, " Average Covered Charges "])
plt.plot(df.loc[:, " Average Medicare Payments "])
plt.show()
"""
