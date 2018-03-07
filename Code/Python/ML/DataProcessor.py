import numpy as np

class DataProcessor(object):
    def __init__(self):
        pass

    def fivefold(self, data , key):
        """
        :param: data is whole data for trainning and testing
        :return: seperate data for five dataframe or list
        """
        self.data = data
        data_five = []
        for i in range(1,6):
            data_five.append(data[data.loc[:, key]==i])
            # print(data_five[i-1])
        return data_five

    def divide_xy(self, data_five , key):
        """
       :return: X and Y data
        """
        self.data = data_five
        y_data_five = []
        x_data_five = []
        for data in data_five :
            y_data_five.append(data.loc[:,key].as_matrix())
            x_data_five.append(data.iloc[:,1:-2].as_matrix())
        return x_data_five , y_data_five


    def train_test(self, datas, key):
        """
        :param: data is whole data for trainning and testing
        :return: seperate data for five dataframe or list
        """
        test_data = datas[key]
        keychecker = 0
        count = 0
        for data in datas :
            if keychecker == key:
                pass
            else :
                if count == 0 :
                    train_data = data
                else :
                    train_data = np.concatenate( (train_data, data ), axis=0)
                count +=1
            keychecker += 1

        return train_data, test_data


    def one_hot_encoder(self, data):
        result = np.zeros((len(data), 2))
        result[np.arange(len(data)), data] = 1

        return result

    def calibration(self, data):
        cal_data = data[:len(data)//2,:]
        test_data = data[len(data)//2:, :]

        return cal_data, test_data