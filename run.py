from util.model import model
from util.data_process_real import Data_Model
import numpy as np
from keras import backend as K
import argparse

save_root = "save_noSG"

models = {0:"TCN", 1:"LSTM", 2:"BiLSTM"}
categories = {0:"Worker", 1:"Business", 2:"Economic", 3:"Sponsor", 4:"Refugee", 5:"Total"}


def error_mse(prediction, truth):
    difference = prediction-truth
    squared = difference **2
    return np.sum(squared)/len(squared)

def error_r2(prediction, truth):
    from sklearn.metrics import r2_score
    return r2_score(truth,prediction)

def error_rmsle(prediction, truth):
    p = np.log(prediction+1)
    t = np.log(truth+1)
    difference = (p-t)
    squared = difference**2

    return np.sqrt(np.sum(squared)/len(squared))

def calculate_performance(model_type, model_name, data_set, prediction, truth):
    np.save(f'{save_root}//{model_type}//{model_name}_predict_{data_set}', prediction)
    np.save(f'{save_root}//{model_type}//{model_name}_truth_{data_set}', truth)


    mse = error_mse(prediction,truth)
    r2 = error_r2(prediction,truth)
    rmsle = error_rmsle(prediction,truth)
    print(f"{data_set} t: {model_type}, n: {model_name}, mse: {mse}, rmsle: {rmsle:>10}, rmsle: {r2:2.20f}")

    f = open(f'{save_root}//acc.txt', 'a')
    f.write(f"{data_set:>10} t: {model_type:>10}, n: {model_name:>10}, mse: {mse:2.20f}, rmsle: {rmsle:2.20f}, r2: {r2:2.20f}\n")
    f.close()

if __name__ == "__main__":
    _argparser = argparse.ArgumentParser(
            description='Recognition',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--timestep', type=int, default=60, metavar='INTEGER',
        help='Time step in network')
    _argparser.add_argument(
        '--type', type=int, default=1, metavar='INTEGER',
        help='model')
    _argparser.add_argument(
        '--sensor', type=int, default=0, metavar='INTEGER',
        help='model')
    _args = _argparser.parse_args()

    model_type = _args.type
    batch_size = 16
    num_frame = _args.timestep
    root_path = r"data_csv//*"

    data_structure = Data_Model(root_path, num_frame=num_frame, skip=np.int(1), sensor=-1)
    total_data, total_label = data_structure.get_data()

    for sensor in range(0, 6):

        data = total_data[:,:,:,[sensor]]
        label = total_label[:,:,[sensor]]

        train_data = data[0][:20]
        train_label = label[0][:20]

        val_data = data[0][20:30]
        val_label = label[0][20:30]

        test_data = data[0][30:]
        test_label = label[0][30:]

        for index in range(1,len(label)):
            train_data = np.vstack((train_data,data[index][:20]))
            train_label = np.vstack((train_label,label[index][:20]))

            val_data = np.vstack((val_data,data[index][20:30]))
            val_label = np.vstack((val_label,label[index][20:30]))

            test_data = np.vstack((test_data,data[index][30:]))
            test_label = np.vstack((test_label,label[index][30:]))

        for model_type in range(0,3):
            model_name = categories[sensor]

            t_model = model(num_classes=1,
                            model_type=(models[model_type],model_name),
                            lr=1e-3,
                            num_frame=num_frame,
                            feature_size=(1,))

            save_file = t_model.train(train_data,
                                      train_label,
                                      val_data,
                                      val_label,
                                      batch_size,
                                      base_epoch=9999,
                                      path=save_root)

            t_model.load(save_file)
            pred_test_label = t_model.predict(np.array(test_data))
            pred_val_label = t_model.predict(np.array(val_data))

            calculate_performance(models[model_type], model_name, "test", pred_test_label,test_label)
            calculate_performance(models[model_type], model_name, "val", pred_val_label,val_label)

            del t_model
            K.clear_session()
