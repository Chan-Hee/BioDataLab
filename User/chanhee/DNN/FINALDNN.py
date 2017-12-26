##################FINAL DNN CODE#######################


##################Import Modules#######################
import random
import numpy as np
import math
import pandas as pd
import tensorflow as tf
tf.set_random_seed(777)


##################Define Functions#####################
def five_fold(data, i):
    test_data = data[data['index']==i+1]
    train_data = data[(data['index']<i+1) | (data['index']>i+1)]
    print(len(test_data), len(train_data))

    return train_data , test_data

def set_train_three_layer(num,repeat, nodes, learning_rate):
    batch_size = 1000
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32)
    train_a = 0
    test_a = 0
    X = tf.placeholder(tf.float32, [None, cnt_train])
    Y = tf.placeholder(tf.float32, [None, 2])

    W1 = tf.get_variable( shape= [cnt_train, nodes[0]], name='weight1' , initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([nodes[0]]), name='bias1')
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

    W2 = tf.get_variable(shape =[nodes[0], nodes[1]], name='weight2', initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([nodes[1]]), name='bias2')
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    layer2 = tf.nn.dropout(layer2 , keep_prob=keep_prob)

    W3 = tf.get_variable(shape= [nodes[1], nodes[2]], name='weight3',initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([nodes[2]]), name='bias3')
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
    layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

    W4 = tf.get_variable(shape=[nodes[2], 2], name='weight4',initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([2]), name='bias4')
    hypothesis = tf.matmul(layer3, W4) + b4



    # cost/loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    # Accuracy computation
    # True if hypothesis>0.5 else False

    predicted = tf.argmax(hypothesis,1)
    correct_prediction = tf.equal(predicted,tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(repeat):
            avg_cost = 0
            total_num = int(len(train_x)/batch_size)

            for i in range(total_num):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                sess.run( train , feed_dict={X: batch_x, Y: batch_y , keep_prob : 0.7})

            if step == repeat-1:
                ####Train Accuracy report####
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y, keep_prob :0.7})
                print("\nTrain Accuracy: ", train_a)
            if step % 20 == 0 :
                h,c, p,train_a = sess.run([hypothesis, cost ,predicted, accuracy],feed_dict={X: train_x, Y: train_y, keep_prob :0.7})
                print("\nCurrent Accuracy : ", train_a , "cost : ", c , "Current Step : ", step)
                if train_a > 0.97 :
                    break
        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y, keep_prob :1.0})
        print("\nTest Accuracy: ", test_a)

    return train_a, test_a


def set_train_four_layer(num ,repeat, nodes, learning_rate):
    batch_size = 1000
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32)

    train_a = 0
    test_a = 0
    X = tf.placeholder(tf.float32, [None, cnt_train])
    Y = tf.placeholder(tf.float32, [None, 2])

    W1 = tf.get_variable( shape= [cnt_train, nodes[0]], name='Weight1' , initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([nodes[0]]), name='Bias1')
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

    W2 = tf.get_variable(shape =[nodes[0], nodes[1]], name='Weight2', initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([nodes[1]]), name='Bias2')
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

    W3 = tf.get_variable(shape= [nodes[1], nodes[2]], name='Weight3',initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([nodes[2]]), name='Bias3')
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
    layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

    W4 = tf.get_variable(shape = [nodes[2], nodes[3]] , name='Weight4' , initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([nodes[3]]), name='Bias4')
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
    layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

    W5 = tf.get_variable(shape = [nodes[3], 2],name='Weight5',initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([2]), name='Bias5')
    hypothesis = tf.matmul(layer4, W5) + b5

    # cost/loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.argmax(hypothesis,1)
    correct_prediction = tf.equal(predicted,tf.argmax(Y,1))


    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(repeat):
            avg_cost = 0
            total_num = int(len(train_x)/batch_size)

            for i in range(total_num):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                sess.run(train, feed_dict={X: batch_x, Y: batch_y , keep_prob : 0.7})

            if step == repeat-1:
                ####Train Accuracy report####
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y, keep_prob :0.7})
                print("\nTrain Accuracy: ", train_a)
            if step % 20 == 0 :
                h,c, p,train_a = sess.run([hypothesis, cost ,predicted, accuracy],feed_dict={X: train_x, Y: train_y, keep_prob :0.7})
                print("\nCurrent Accuracy : ", train_a , "cost : ", c , "Current Step : ", step)
                if train_a > 0.97 :
                    break

        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y, keep_prob :1.0})
        print("\nTest Accuracy: ", test_a)

    return train_a, test_a
##################READ DATA############################

datafilename = "/home/tjahn/Data/FinalData.csv"
data = pd.read_csv(datafilename)
conf_filename = '/home/tjahn/Git/Data/input/relu_test_ps8.csv'
conf = pd.read_csv(conffilename)
##################TRAIN MODEL AS CONF#################
train_accs = []
test_accs = []
for i in range(len(conf)):
    repeat, layer, node , learning_rate, gene = conf.iloc[i]
    nodes = list(map(int , node.split(" ")))
    train_accs_conf = []
    test_accs_conf = []
    for j in range(5):
        #####Five fold#####
        train_data, test_data = five_fold(data, j)
        cal_data = test_data[:int(len(test_data.columns)/2),:]
        test_data = test_data[int(len(test_data.columns)/2):,] 

        #####Train Data Set#####
        train_x = train_data.iloc[:,:-2]
        train_x = train_x.as_matrix()
        train_y = train_data.iloc[:,-2].as_matrix()
        train_y = train_y.flatten()
        train_y = pd.get_dummies(train_y)

        #####Test Data Set#####
        test_x = test_data.iloc[:,:-2]
        test_x = test_x.as_matrix()
        test_y = test_data.iloc[:,-2].as_matrix()
        test_y = test_y.flatten()
        test_y = pd.get_dummies(test_y)

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

##################STORE RESULT##########################
train_accs = pd.DataFrame(data=train_accs ,
                          index = list(range(len(conf))) ,
                          columns = ["tr-fold-1","tr-fold-2","tr-fold-3","tr-fold-4","tr-fold-5"])
test_accs = pd.DataFrame(data=test_accs ,
                          index = list(range(len(conf))) ,
                          columns = ["te-fold-1","te-fold-2","te-fold-3","te-fold-4","te-fold-5"])

accuracies = pd.concat([train_accs, test_accs], axis=1)
conf = pd.concat([conf, accuracies] , axis = 1)
conf.to_csv( conf_directory+'output'+conf_filename[5:-4] +'_result.csv' , sep= ',')
