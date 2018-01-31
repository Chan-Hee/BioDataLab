from DeepNeuralNetwork import DNN
from ConvolutionNeuralNetwork import  CNN
from DataProcessor import DataProcessor
import pandas as pd
import numpy as np


input_directory = "/Users/taewan/Desktop/"
output_directory = "/Users/taewan/Desktop/Ahn/"


def main() :
    try : selectmodel = int(input("1. Deep Neural Network\n"
                        "2. Convolutional Neural Network\n"
                        "Choose Model : "))
    except :
        selectmodel = 10

    while selectmodel != 1 and selectmodel != 2:
        print("Wrong number of model, Please Insert number again.")
        selectmodel = int(input("1. Deep Neural Network\n"
                                "2. Convolutional Neural Network\n"
                                "Choose Model : "))

    #Process Data#
    data = pd.read_csv(input_directory+"FinalTest_data.csv")
    dataProcessor = DataProcessor()
    data_fivefold = dataProcessor.fivefold(data, 'index')
    x_data_five, y_data_five = dataProcessor.divide_xy(data_fivefold, 'result')
    train_x, test_x = dataProcessor.train_test(x_data_five, 0)
    train_y, test_y = dataProcessor.train_test(y_data_five, 0)
    train_y = dataProcessor.one_hot_encoder(train_y)
    test_y = dataProcessor.one_hot_encoder(test_y)
    cal_x , test_x = dataProcessor.calibration(test_x)
    cal_y , test_y = dataProcessor.calibration(test_y)


    if selectmodel == 1 :
        model = DNN([1000,1000,1000])
        model.fit(train_x , train_y, cal_x, cal_y)
        model.test(test_x, test_y)

    else:
        model = CNN([10,10,10])

    print(model.get_name())


main()