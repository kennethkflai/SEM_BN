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

def extract_time(data, num_frame, skip, sensor):

    class_label = data[:,-1]
    
    if sensor == -1:
        temp_data = data[:,:-1]
    else:
        temp_data = data[:,sensor]

    data = [temp_data[i:i+num_frame] for i in range(0,len(temp_data)-num_frame,skip)]
    label = [class_label[i+num_frame] for i in range(0,len(temp_data)-num_frame,skip)]


    return data, label


class Data_Model(object):
    """
    Class used for loading and processing data
    """
    def __init__(self,
        root_path,                 # Directory of the location of the dataset
        num_frame=11,
        skip=1,
        sensor=0
        ):

        if skip <1:
            skip = 1

        file_list = glob(root_path, recursive=True)

        self.train_label = []
        self.train_data = []
        self.test_label = []
        self.test_data = []
        
        for index in tqdm(range(len(file_list))):
            df = pd.read_csv(file_list[index])
        
            data = df.iloc[:,4:]
            data = np.array(data)
            
            if np.isnan(data).any():
                
                print(file_list[index])
                nan_index = np.argwhere(np.isnan(data))
                for nindex in range(len(nan_index)):
                    data = np.delete(data,[nan_index[len(nan_index)-1-nindex][0]],0)

                print(nan_index[0])

                
            general_data = normalize_data(data,use_sg_filter=False)
            data, label = extract_time(data, num_frame, skip, sensor)
        
            if "train" in file_list[index]:
                self.train_data.append(data)
                self.train_label.append(label)
                
            else:
                self.test_data.append(data)
                self.test_label.append(label)
        
    def get_data(self):
        return self.train_data, self.train_label, self.test_data, self.test_label

if __name__ == "__main__":

    root_path = r"../data_csv//*//*"

    file_list = glob(root_path, recursive=True)
    data_structure = Data_Model(root_path, num_frame=11, skip=np.int(1), sensor=-1)
    train_data, train_label, test_data, test_label = data_structure.get_data()
    

    
    
    