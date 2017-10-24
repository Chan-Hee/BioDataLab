####Import Modules####
import tensorflow as tf
import random
import numpy as np
import math
import pandas as pd

####Set Random Seed ####
tf.set_random_seed(777) 

def cal_var(variances, per):
    all_cnt = len(variances)
    per = 100-per
    per_idx = int(all_cnt*(per/100))
    return variances[per_idx]
    
def five_fold(data, num):
    test_set = data[num*2000:(num+1)*2000]
    train_set = data[:num*2000]
    train_set = np.concatenate((train_set,data[(num+1)*2000:] ), axis=0)
    return train_set , test_set
    
def top_of_variance(per,  data_x):
    ##data_x['variance']
    ##calculate value  
    data = data_x[data_x['VAR'] > per]
    idx_list = data.index.tolist()
    ##return index 
    return idx_list

def set_train_three_layer(num,repeat, nodes, learning_rate):
    train_a = 0
    test_a = 0
    X = tf.placeholder(tf.float32, [None, cnt_train])
    Y = tf.placeholder(tf.float32, [None, 1])

    W1 = tf.Variable(tf.random_normal([cnt_train, nodes[0]]), name='weight1')
    b1 = tf.Variable(tf.random_normal([nodes[0]]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([nodes[0], nodes[1]]), name='weight2')
    b2 = tf.Variable(tf.random_normal([nodes[1]]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    W3 = tf.Variable(tf.random_normal([nodes[1], nodes[2]]), name='weight3')
    b3 = tf.Variable(tf.random_normal([nodes[2]]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

    W4 = tf.Variable(tf.random_normal([nodes[2], 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(repeat):
            sess.run(train, feed_dict={X: train_x, Y: train_y})
            if step == repeat-1:
                ####Train Accuracy report####
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y})
                print("\nTrain Accuracy: ", train_a)
            if step % 2000 == 0 : 
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y})
                print("\nCurrent Accuracy : ", train_a , "Current Step : ", step)
        ######Accuracy Report#####
        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y})    
        print("\nTest Accuracy: ", test_a)
    
    return train_a, test_a
        
def set_train_four_layer(num ,repeat, nodes, learning_rate):
    train_a = 0
    test_a = 0
    X = tf.placeholder(tf.float32, [None, cnt_train])
    Y = tf.placeholder(tf.float32, [None, 1])

    W1 = tf.Variable(tf.random_normal([cnt_train, nodes[0]]), name='weight1')
    b1 = tf.Variable(tf.random_normal([nodes[0]]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([nodes[0], nodes[1]]), name='weight2')
    b2 = tf.Variable(tf.random_normal([nodes[1]]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    W3 = tf.Variable(tf.random_normal([nodes[1], nodes[2]]), name='weight3')
    b3 = tf.Variable(tf.random_normal([nodes[2]]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

    W4 = tf.Variable(tf.random_normal([nodes[2], nodes[3]]), name='weight4')
    b4 = tf.Variable(tf.random_normal([nodes[3]]), name='bias4')
    layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

    W5 = tf.Variable(tf.random_normal([nodes[3], 1]), name='weight5')
    b5 = tf.Variable(tf.random_normal([1]), name='bias5')
    hypothesis = tf.sigmoid(tf.matmul(layer4, W5) + b5)

    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(repeat):
            sess.run(train, feed_dict={X: train_x, Y: train_y})
            if step == repeat-1:
                ####Train Accuracy report####
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y})
                print("\nTrain Accuracy: ", train_a)
            if step % 2000 == 0 : 
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y})
                print("\nCurrent Accuracy : ", train_a , "Current Step : ", step)
        ######Accuracy Report#####
        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y})    
        print("\nTest Accuracy: ", test_a)
    
    return train_a, test_a

####Read data####

x_filename = '/home/tjahn/Data/DNN10000/DNN10000.csv'
xdata = pd.read_csv(x_filename)
ydata = np.genfromtxt('/home/tjahn/Data/DNN10000/CancerResult.csv', delimiter=",")

conf_filename = '/home/tjahn/learning_ps.csv'
conf = pd.read_csv(conf_filename)
#print(conf)
train_accs = []
test_accs = []
for i in range(len(conf)):
    repeat, layer, node, learning_rate, gene = conf.iloc[i]
    nodes = list(map(int, node.split(" ")))
    train_accs_conf = []
    test_accs_conf = []
    for j in range(5):
        variance_set = pd.concat([xdata.iloc[:2000*j], xdata.iloc[2000*(j+1):]])
        #print(variance_set.iloc[:,-1])
        variances = variance_set.iloc[:, -1]
        #print(variances)
        variances = variances.as_matrix()
        #print(variances)
        variances = np.sort(variances)
        #print(variances)
        idx = top_of_variance(cal_var(variances, gene), variance_set)
        data_x = xdata.loc[idx]
        data_x = data_x.as_matrix()
        data_x = data_x[1:, 3:-1]
        data_y = ydata[1:, 1:]    # eliminate heading, string data
        data_x = data_x.transpose()
        ###5-fold data 
        ####Data Processing - divide train and test set####
        ####5-fold code  is needed ####
        print(len(data_x), len(data_y))
        train_x, test_x = five_fold(data_x,j)
        train_y, test_y = five_fold(data_y,j)
        #print(train_y)
        cnt_train = len(train_x[1, :])
        if(conf.iloc[i]['layer'] == 3):
            train_acc , test_acc = (set_train_three_layer(i,repeat, nodes, learning_rate))
            train_accs_conf.append(train_acc)
            test_accs_conf.append(test_acc)
        elif(conf.iloc[i]['layer']== 4):
            train_acc , test_acc = (set_train_four_layer(i,repeat, nodes, learning_rate))
            train_accs_conf.append(train_acc)
            test_accs_conf.append(test_acc)
    train_accs.append(train_accs_conf)
    test_accs.append(test_accs_conf)


train_accs = pd.DataFrame(data=train_accs , 
                          index = list(range(len(conf))) , 
                          columns = ["tr-fold-1","tr-fold-2","tr-fold-3","tr-fold-4","tr-fold-5"])
test_accs = pd.DataFrame(data=test_accs , 
                          index = list(range(len(conf))) , 
                          columns = ["te-fold-1","te-fold-2","te-fold-3","te-fold-4","te-fold-5"])

accuracies = pd.concat([train_accs, test_accs], axis=1)
conf = pd.concat([conf, accuracies] , axis = 1)
conf.to_csv(conf_filename[:-4]+'result.csv')
