import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename = "input_data.csv"
df = pd.read_csv(filename)

class DRG(object):
    def __init__(self):
        self.DRG         = df.iloc[:, 0].tolist()
        self.idxDRG      = df.iloc[:, 0].unique().tolist()
        self.idxDRG_num  = np.zeros(len(self.idxDRG))
        self.DRG_num     = np.zeros(len(self.DRG))
        for i in range(len(self.idxDRG)):
            self.idxDRG_num[i]   = self.idxDRG[i].split(' ')[0].split("'")[0]
        for i in range(len(self.DRG)):
            self.DRG_num[i]  = self.DRG[i].split(' ')[0].split("'")[0]
        self.idxDRG_num  = self.idxDRG_num.astype(int)
        self.DRG_num = self.DRG_num.astype(int)

        self.ProvID = df.iloc[:, 1].tolist()
        #return self.idxDRG_num, self.idxDRG, self.DRG_num, self.DRG, self.ProvID

    def __getitem__(self):
        return self.DRG, self.idxDRG, self.DRG_num, self.idxDRG_num, self.ProvID

