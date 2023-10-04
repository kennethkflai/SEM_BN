from util.model import model
from util.data_process import Data_Model
import numpy as np
from keras import backend as K
import argparse
from sklearn.metrics import classification_report

save_root = "save"

models = {0:"TCN", 1:"LSTM", 2:"BiLSTM"}
categories = {0:"HR", 1:"RR", 2:"GSR", 3:"Temp"}


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
    root_path = r"data_csv//*//*"


    data_structure = Data_Model(root_path, num_frame=num_frame, skip=np.int(1), sensor=-1)
    subject_train_data, subject_train_label, subject_test_data, subject_test_label = data_structure.get_data()

    train_label = subject_train_label[0]
    train_data_total = subject_train_data[0]
    for i in range(1,len(subject_train_label)):
        train_label = train_label + subject_train_label[i]
        train_data_total = train_data_total + subject_train_data[i]
        
    test_label = subject_test_label[0]
    test_data_total = subject_test_data[0]
    for i in range(1,len(subject_test_label)):
        test_label = test_label + subject_test_label[i]
        test_data_total = test_data_total + subject_test_data[i]
          

    for sensor in range(0, 4):
        test_data = np.array(test_data_total)[:,:,sensor]
        test_data = test_data[:,:,np.newaxis]
        train_data = np.array(train_data_total)[:,:,sensor]
#        train_data = np.random.random((54642,11))
        train_data = train_data[:,:,np.newaxis]
        
        for model_type in range(0,3):
            model_name = categories[sensor]

            t_model = model(num_classes=len(np.unique(train_label)),
                            model_type=(models[model_type],model_name),
                            lr=1e-3,
                            num_frame=num_frame,
                            feature_size=(1,))

            save_file = t_model.train(train_data,
                                      train_label,
                                      test_data,
                                      test_label,
                                      batch_size,
                                      base_epoch=9999,
                                      path=save_root)

            t_model.load(save_file)
            pred_test_label = t_model.predict(np.array(test_data))
            prediction = np.argmax(pred_test_label,1)
# t_model.predict(np.ones((1,11,1)))
            np.save(f'{save_root}//{models[model_type]}//{model_name}_predict_test', prediction)
            np.save(f'{save_root}//{models[model_type]}//{model_name}_truth_test', test_label)
        
        
            report = classification_report(test_label, prediction,output_dict=True)
            f = open(f'{save_root}//acc.txt', 'a')
                        
            f.write(f'''TS: {num_frame:3.0f}, model: {models[model_type]:5s}, sensor: {model_name}, classes: {len(np.unique(train_label))}, BatchSize: {batch_size:3d}, Accuracy: {report['accuracy']:.8f}, f1: {report['macro avg']['f1-score']:.8f}, precision: {report['macro avg']['precision']:.8f}, recall: {report['macro avg']['recall']:.8f} \n''')
            
            f.close()
            
            del t_model
            K.clear_session()
