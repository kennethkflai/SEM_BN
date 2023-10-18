from glob import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.signal import savgol_filter

def max_min_scaler(data):
    '''
    Rescale data to [0,1] using the min and max value
    '''
    mx = np.max(data)
    mn = np.min(data)

    if mx-mn == 0:
        scaled = np.zeros((len(data),))
    else:
        scaled = (data-mn)/(mx-mn)

    return scaled

def standardize(data):
    '''
    Standardize data such that the mean of data is 0 and standard deviation is 1
    '''
    average = np.mean(data)
    deviation = np.std(data)
    
    standard = (data-average)/deviation
    
    return standard
    
def preprocess_data(data,
                    use_sg_filter=True,
                    use_scaling=True,
                    use_standard=True):
    '''
    Preprocess data by:
        1) smoothing via savgol filter
        2) scaling data to [0,1]
        3) standardize data so mean=0, std=1
    '''
    processed_data = []
    for column in range(len(data[0])):
        temp_data = data[:,column]

        if use_sg_filter == True:
            temp_data = savgol_filter(temp_data,25,3) #parameters 25, 3, nearest
            
        if use_scaling == True:
            temp_data = max_min_scaler(temp_data)
            
        if use_standard == True:
            temp_data = standardize(temp_data)
            
        processed_data.append(temp_data)

    return np.swapaxes(np.array(processed_data),0,1)

def segments(data, labels):
    '''
    Separate data into individual task segments.  Each data file contains ~41 tasks where the last 30 second is of interest
    '''
    temp_data = []
    label = []
    
    temp_label = labels[0]
    initial = 0
    
    for i in range(len(labels)):
        if temp_label != labels[i]:
            temp_data.append(data[initial:i])
            initial = i
            label.append(labels[i-1])
            
            temp_label = labels[i]
            
    return temp_data, label
    
def extract_time(data, labels, num_frame, skip, sensor):
    '''
    For each task, use sliding window to extract time frame based on window size and skip
    '''

    if sensor == -1:
        temp_data = data
    else:
        temp_data = data[:,sensor]

    temp_data, labels = segments(temp_data, labels)
    
    data = []
    label = []
    for index,segment in enumerate(temp_data):
        
        data.append([segment[i:i+num_frame] for i in range(0,len(segment)-num_frame,skip)])
        label.append(labels[index])
        
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
        
            data = df.iloc[:,5:9]
            labels = np.array(df.iloc[:,4])
            data = np.array(data)
            
            if np.isnan(data).any():
                nan_index = np.argwhere(np.isnan(data))
                for nindex in range(len(nan_index)):
                    data = np.delete(data,[nan_index[len(nan_index)-1-nindex][0]],0)
                    labels = np.delete(labels,[nan_index[len(nan_index)-1-nindex][0]],0)
                    
            general_data = preprocess_data(data,use_sg_filter=True)
            data, label = extract_time(general_data, labels, num_frame, skip, sensor)
            
        
            if "train" in file_list[index]:
                self.train_data.append(data)
                self.train_label.append(label)
                
            else:
                self.test_data.append(data)
                self.test_label.append(label)
        
    def get_data(self):
        return self.train_data, self.train_label, self.test_data, self.test_label

if __name__ == "__main__":

    root_path = r"../data//*//*"

    file_list = glob(root_path, recursive=True)
    data_structure = Data_Model(root_path, num_frame=11, skip=np.int(1), sensor=-1)
    train_data, train_label, test_data, test_label = data_structure.get_data()
    

    
    
    