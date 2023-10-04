from glob import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.signal import savgol_filter

def max_min_scaler(data):
    mx = np.max(data)
    mn = np.min(data)

    if mx-mn == 0:
        scaled = np.zeros((len(data),))
    else:
        scaled = (data-mn)/(mx-mn)

#    print(len(data))
    return scaled

def normalize_data(data,use_sg_filter=True):

    # natural logarithm
    # apply sg filter
    # apply max min scaler

#    data = np.log(data)

    normalized_data = []
    for column in range(len(data[0])):
        temp_data = data[:,column]

        if use_sg_filter == True:
            temp_data = savgol_filter(temp_data,11,6)

        temp_data = max_min_scaler(temp_data)
        normalized_data.append(temp_data)

    return np.swapaxes(np.array(normalized_data),0,1)

def extract_time(data,num_frame, skip,sensor):

    if sensor == -1:
        temp_data = data
    else:
        temp_data = data[:,sensor]

    dat = [temp_data[i:i+num_frame] for i in range(0,len(temp_data)-num_frame,skip)]
    label = [temp_data[i+num_frame] for i in range(0,len(temp_data)-num_frame,skip)]


    return dat, label

def extract_time2(data,num_frame, skip,sensor):
    if sensor == -1:
        temp_data = data
    else:
        temp_data = data[:,sensor]

    dat = []
    label = []
    temp =  list(temp_data[:num_frame])
    dat.append(temp)
    label.append(temp_data[num_frame])

    for i in range(num_frame,len(temp_data)-num_frame-1,skip):
        temp.append(temp_data[i])
        temp.pop(0)

        dat.append(temp.copy())

        label.append(temp_data[i+1])

    return dat, label


class Data_Model(object):
    """
    Class used for loading and processing data
    """
    def __init__(self,
        root_path,                 # Directory of the location of the dataset
        num_frame=60,
        skip=1,
        sensor=0
        ):

        if skip <1:
            skip = 1

        file_list = glob(root_path, recursive=True)

        self.label = []
        self.data = []
        self.province_label = []
        for index in tqdm(range(len(file_list))):
            df = pd.read_csv(file_list[index])

            data = np.array(df)[:,1:]

            general_data = normalize_data(data,use_sg_filter=False)

            data, label = extract_time(general_data,num_frame, skip,sensor)
#            data, label = extract_time2(general_data,num_frame, skip,sensor)
            self.data.append(data)
            self.label.append(label)

            province_label = file_list[index]
            province_label = province_label[province_label.rfind("\\")+1:-4]
            self.province_label.append(province_label)


        self.data = np.array(self.data)
        self.label = np.array(self.label)
    def get_data(self):
        return self.data, self.label

if __name__ == "__main__":

    file = r"../data_csv//*"
    from time import time

    st = time()
    data_model = Data_Model(file)
    data,label = data_model.get_data()
    print(time()-st)

#    r = np.random.rand(100)

#    a = max_min_scaler(r)