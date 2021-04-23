import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#default
filename = 'input_data.csv'
df = pd.read_csv(filename)

print(df.columns)
plt.scatter(x= [k for k in range(len(df))], y=df.iloc[:, -4])
plt.show()
