from util.model import model
from util.data_process import Data_Model
import numpy as np
from keras import backend as K
import argparse
from sklearn.metrics import classification_report

save_root = "save"

models = {0:"TCN", 1:"LSTM", 2:"BiLSTM", 3:"TCN+LSTM"}
categories = {-1:"All", 0:"HR", 1:"RR", 2:"GSR", 3:"Temp"}


if __name__ == "__main__":
    _argparser = argparse.ArgumentParser(
            description='Recognition',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--timestep', type=int, default=11, metavar='INTEGER',
        help='Time step in network')
    _argparser.add_argument(
        '--sensor', type=int, default=0, metavar='INTEGER',
        help='model')
    _args = _argparser.parse_args()

    batch_size = 256
    num_frame = _args.timestep
    root_path =  r"data//*//*"

    for fr in range(3,7):
        num_frame = 2**fr
        skip = 1#max(np.int(1), np.int(num_frame//2))
        
        data_structure = Data_Model(root_path, num_frame=num_frame, skip=skip, sensor=-1)
        subject_train_data, subject_train_label, subject_test_data, subject_test_label = data_structure.get_data()
    
        segment_train_label = subject_train_label[0]
        segment_train_data = subject_train_data[0]
        for i in range(1,len(subject_train_label)):
            segment_train_label = segment_train_label + subject_train_label[i]
            segment_train_data = segment_train_data + subject_train_data[i]
            
        train_label = list([segment_train_label[0] for i in range(len(segment_train_data[0]))])
        train_data_total = segment_train_data[0]
        for i in range(1,len(segment_train_label)):
            train_label = train_label + list([segment_train_label[i] for j in range(len(segment_train_data[i]))])
            train_data_total = train_data_total + segment_train_data[i]
            
        segment_test_label = subject_test_label[0]
        segment_test_data = subject_test_data[0]
        for i in range(1,len(subject_test_label)):
            segment_test_label = segment_test_label + subject_test_label[i]
            segment_test_data = segment_test_data + subject_test_data[i]
            
        test_label = list([segment_test_label[0] for i in range(len(segment_test_data[0]))])
        test_data_total = segment_test_data[0]
        for i in range(1,len(segment_test_label)):
            test_label = test_label + list([segment_test_label[i] for j in range(len(segment_test_data[i]))])
            test_data_total = test_data_total + segment_test_data[i]
            
        classes = {lbl:index for index,lbl in enumerate(np.unique(train_label))}
        resting = {lbl:"est" in lbl for index,lbl in enumerate(np.unique(train_label))}
        
    #    for index, lbl in enumerate(train_label):
    #        lbl = classes[lbl]
    #        train_label[index] = lbl
            
        train_label = [resting[lbl] for lbl in train_label]
        test_label = [resting[lbl] for lbl in test_label]
    
#        train_label = [classes[lbl] for lbl in train_label]
#        test_label = [classes[lbl] for lbl in test_label]
        
        for sensor in range(-1, 4):
            if sensor==-1:
                test_data = np.array(test_data_total)
                train_data = np.array(train_data_total)
            else: 
                test_data = np.array(test_data_total)[:,:,sensor]
                test_data = test_data[:,:,np.newaxis]
                
                train_data = np.array(train_data_total)[:,:,sensor]
                train_data = train_data[:,:,np.newaxis]
            
            for model_type in range(0,4):
                model_name =f"{categories[sensor]}_frame{num_frame}_skip{skip}"
    
                t_model = model(num_classes=len(np.unique(train_label)),
                                model_type=(models[model_type],model_name),
                                lr=1e-4,
                                num_frame=num_frame,
                                feature_size=(train_data.shape[2],))
    
                save_file = t_model.train(train_data,
                                          train_label,
                                          test_data,
                                          test_label,
                                          batch_size,
                                          base_epoch=9999,
                                          path=save_root)
    
                t_model.load(save_file)
                pred_test_label = t_model.predict(np.array(test_data))
                
                if model_type == 3:
                    prediction = np.argmax(pred_test_label[-1],1)
                else:
                    prediction = np.argmax(pred_test_label,1)
                    
                np.save(f'{save_root}//{models[model_type]}//{model_name}_predict_test', prediction)
                np.save(f'{save_root}//{models[model_type]}//{model_name}_truth_test', test_label)
                
                report = classification_report(test_label, prediction,output_dict=True)
                f = open(f'{save_root}//acc.txt', 'a')
                            
                f.write(f'''TS: {num_frame:3.0f}, skip: {skip}, model: {models[model_type]:5s}, sensor: {model_name}, classes: {len(np.unique(train_label))}, BatchSize: {batch_size:3d}, Accuracy: {report['accuracy']:.8f}, f1: {report['macro avg']['f1-score']:.8f}, precision: {report['macro avg']['precision']:.8f}, recall: {report['macro avg']['recall']:.8f} \n''')
                
                f.close()
                
                del t_model
                K.clear_session()
