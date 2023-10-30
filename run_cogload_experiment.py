from util.model import model
from util.data_process import Data_Model
import numpy as np
from keras import backend as K
import argparse
from sklearn.metrics import classification_report

save_root = "save"

models = {0:"TCN", 1:"LSTM", 2:"BiLSTM", 3:"TCN+LSTM", 4:"TCN+BiLSTM"}
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

    batch_size = 32
    num_frame = _args.timestep
    root_path =  r"data//*//*"

    num_frame = 30
    skip = 1#max(np.int(1), np.int(num_frame//2))
    
    data_structure = Data_Model(root_path, num_frame=num_frame, skip=skip, sensor=-1)
    subject_train_data, subject_train_label, subject_test_data, subject_test_label = data_structure.get_data()
    
    for cv in range(0,3):
        train_label = []
        train_data_total = []
        val_label = []
        val_data_total = []
        test_label = []
        test_data_total = []
        for i in range(len(subject_test_label)):
            for j in range(len(subject_test_label[i])):
                if len(subject_test_data[i][j]) != 0:
                    test_label.append(subject_test_label[i][j])
                    test_data_total.append(subject_test_data[i][j][-1])
                
        for i in range(len(subject_train_label)):
            for j in range(len(subject_train_label[i])):
                if len(subject_train_data[i][j]) != 0:
                    if i%3 == cv:
                        val_label.append(subject_train_label[i][j])
                        val_data_total.append(subject_train_data[i][j][-1])
                    else:
                        train_label.append(subject_train_label[i][j])
                        train_data_total.append(subject_train_data[i][j][-1])
                
    
        resting = {lbl:"est" in lbl for index,lbl in enumerate(np.unique(train_label))}
        train_label = [resting[lbl] for lbl in train_label]
        test_label = [resting[lbl] for lbl in test_label]
        
        val_label = [resting[lbl] for lbl in val_label]
        
    #    classes = {lbl:index for index,lbl in enumerate(np.unique(train_label))}
    #    train_label = [classes[lbl] for lbl in train_label]
    #    test_label = [classes[lbl] for lbl in test_label]
        
        for sensor in range(3, -2,-1):
            if sensor==-1:
                test_data = np.array(test_data_total)
                train_data = np.array(train_data_total)
                val_data = np.array(val_data_total)
            else: 
                test_data = np.array(test_data_total)[:,:,sensor]
                test_data = test_data[:,:,np.newaxis]
                
                train_data = np.array(train_data_total)[:,:,sensor]
                train_data = train_data[:,:,np.newaxis]
                
                val_data = np.array(val_data_total)[:,:,sensor]
                val_data = val_data[:,:,np.newaxis]
            
            
            for model_type in range(4,5):
                model_name =f"{categories[sensor]}_frame{num_frame}_skip{skip}_cv{cv}"
    
                t_model = model(num_classes=len(np.unique(train_label)),
                                model_type=(models[model_type],model_name),
                                lr=1e-4,
                                num_frame=num_frame,
                                feature_size=(train_data.shape[2],))
    
                save_file = t_model.train(train_data,
                                          train_label,
                                          val_data,
                                          val_label,
                                          batch_size,
                                          base_epoch=9999,
                                          path=save_root)
    
                t_model.load(save_file)
                pred_test_label = t_model.predict(test_data)
                pred_val_label = t_model.predict(val_data)
                
                if model_type == 3 or model_type == 4:
                    prediction = np.argmax(pred_test_label[-1],1)
                    validation = np.argmax(pred_val_label[-1],1)
                else:
                    prediction = np.argmax(pred_test_label,1)
                    validation = np.argmax(pred_val_label,1)
                    
                np.save(f'{save_root}//{models[model_type]}//{model_name}_predict_test', prediction)
                np.save(f'{save_root}//{models[model_type]}//{model_name}_truth_test', test_label)
                
                report = classification_report(test_label, prediction,output_dict=True)
                reportv = classification_report(val_label, validation,output_dict=True)
                
                f = open(f'{save_root}//acc.txt', 'a')
                            
                f.write(f'''TS: {num_frame:3.0f}, skip: {skip}, cv: {cv}, model: {models[model_type]:5s}, sensor: {categories[sensor]}, classes: {len(np.unique(train_label))}, BatchSize: {batch_size:3d}, Val Accuracy: {reportv['accuracy']:.8f}, Accuracy: {report['accuracy']:.8f}, f1: {report['macro avg']['f1-score']:.8f}, precision: {report['macro avg']['precision']:.8f}, recall: {report['macro avg']['recall']:.8f} \n''')
                
                f.close()
                
                del t_model
                K.clear_session()
