####Import Modules####
import tensorflow as tf
import random
import numpy as np
import math
import pandas as pd

####Set Random Seed ####
tf.set_random_seed(777)  

def set_train_three_layer(num,repeat, nodes, learning_rate):
    print(num)
    train_a = 0
    test_a = 0
    repeat = 1
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
        ######Accuracy Report#####
        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y})    
        print("\nTest Accuracy: ", test_a)
    
    return train_a, test_a
        
def set_train_four_layer(num ,repeat, nodes, learning_rate):
    print(num)
    train_a = 0
    test_a = 0
    repeat = 1
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
        ######Accuracy Report#####
        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y})    
        print("\nTest Accuracy: ", test_a)
    
    return train_a, test_a
        

####Read data####
xdata = np.genfromtxt ('/home/tjahn/Data/DNNData/DNNToyVar1.csv', delimiter=",",usecols=np.arange(0,2007))
ydata = np.genfromtxt ('/home/tjahn/Data/DNNData/CancerResult.csv', delimiter=",")
xdata = xdata[1:, 3:-1]  # eliminate heading, string data, variance
ydata = ydata[1:, 1:]    # eliminate heading, string data
xdata = xdata.transpose()

####Data Processing - divide train and test set####
train_x = xdata[:1000]
train_y = ydata[:1000]
test_x = xdata[1000:]
test_y = ydata[1000:]

cnt_train = len(train_x[1, :])

####Read train configuration
filename = input("Insert configuration filename : ")
conf = pd.read_csv(filename)
print(conf)

train_accs = []
test_accs = []
for i in range(len(conf)):
    train_acc = 0
    test_acc = 0
    if(conf.iloc[i]['layer'] == 3):
        repeat, layer, node , learning_rate, gene = conf.iloc[i]
        nodes = list(map(int , node.split(" ")))
        train_acc , test_acc = (set_train_three_layer(i,repeat, nodes, learning_rate))
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    elif(conf.iloc[i]['layer']== 4):
        repeat, layer, node , learning_rate, gene = conf.iloc[i]
        nodes = list(map(int , node.split(" ")))
        train_acc , test_acc = (set_train_four_layer(i,repeat, nodes, learning_rate))
        train_accs.append(train_acc)
        test_accs.append(test_acc)

conf['train_acc'] = train_accs
conf['test_acc'] = test_accs

conf.to_csv('/Home/tjahn/Result.csv')
