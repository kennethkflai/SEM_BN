# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:51:47 2023

@author: BT_Lab
"""
import numpy as np

frames = 64
skip = 1
nodeList = ["RR",
            "HR",
            "Temp",
            "GSR",
            "Physiological Workload"
           ]

modelType = "TCN+LSTM"
#model = "LSTM"
data=[[] for i in range(4)]
for index, p in enumerate(nodeList[:-1]):

    #p = "Temp"
    predict_path = f".\\save_2\\{modelType}\\{p}_frame{frames}_skip{skip}_predict_test.npy"
    prediction = np.load(predict_path)

    #truth_path = f".\\save\\{modelType}\\{p}_frame{frames}_skip{skip}_truth_test.npy"
    #truth = np.load(truth_path)

    rest = prediction==1
    data[index].append(prediction)

data = np.array(data)
data=np.swapaxes(data,1,-1)
data = data[:,:,0]
import pandas as pd

df = pd.DataFrame(np.transpose(data), columns=nodeList[:-1])

for hr in range(2):
    for rr in range(2):
        for gsr in range(2):
            for temp in range(2):   
                temp_df = df[(df["HR"]==hr) & (df["RR"]==rr) & (df["GSR"]==gsr) & (df["Temp"]==temp)]

#                model.cpt("Physiological Workload")[{'RR': rr, 'HR': hr, 'Temp': temp, 'GSR': gsr}]=[len(temp_df),len(df)-len(temp_df)]    
#data = np.delete(data,-1)
