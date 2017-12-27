import random
import numpy as np
import math
import pandas as pd
import tensorflow as tf

tf.set_random_seed(777) 

def CNN():
    tf.set_random_seed(777)  # reproducibility
    tf.reset_default_graph()

    learning_rate = 0.001
    training_epochs = 30
    batch_size = 100

    # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
    keep_prob = tf.placeholder(tf.float32)

    # input place holders
    X = tf.placeholder(tf.float32, [None, 6084])
    X_img = tf.reshape(X, [-1, 78, 78, 1])   # img 28x28x1 (black/white)
    Y = tf.placeholder(tf.float32, [None, 2])

    # L1 ImgIn shape=(?, 7, 96, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    #    Conv     -> (?, 7, 96, 32)
    #    Pool     -> (?, 4, 48, 32)
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    print(L1)
    # L2 ImgIn shape=(?, 14, 14, 32)
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    #    Conv      ->(?, 14, 14, 64)
    #    Pool      ->(?, 7, 7, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    print(L2)

    # L3 ImgIn shape=(?, 7, 7, 64)
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    #    Conv      ->(?, 7, 7, 128)
    #    Pool      ->(?, 4, 4, 128)
    #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    print(L3)
    L3_flat = tf.reshape(L3, [-1, 128 * 100])
    print(L3_flat)
    # L4 FC 4x4x128 inputs -> 625 outputs
    W4 = tf.get_variable("W4", shape=[128 * 100, 1280],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([1280]))
    L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    print(L4)
    W5 = tf.get_variable("W5",shape= [1280, 128] ,initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([128]))
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
    L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

    W6 = tf.get_variable("W6",shape= [128, 12] ,initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([12]))
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
    L6 = tf.nn.dropout(L6, keep_prob=keep_prob)
    
    
    print(L4)
    # L5 Final FC 625 inputs -> 10 outputs
    W7 = tf.get_variable("W7", shape=[12, 2],
                         initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([2]))
    logits = tf.matmul(L6, W7) + b7
    print(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # initialize
    # train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(train_x)/batch_size)

        for i in range(total_batch):
            batch_xs = train_x[i*batch_size:(i+1)*batch_size]
            batch_ys = train_y[i*batch_size:(i+1)*batch_size]
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
                                X: test_x, Y: test_y, keep_prob: 1}))
    
    print('Learning Finished!')
    return 

def five_fold(data, i):
    test_data = data[data['index']==i+1]
    train_data = data[(data['index']<i+1) | (data['index']>i+1)]
    print(len(test_data), len(train_data))

    return train_data , test_data

#data = pd.read_csv("/Users/Taewan/Desktop/Ahn/FinalData_GSM_gene_index_result.csv")
data = pd.read_csv("/home/tjahn/Data/FinalData_GSM_gene_index_result.csv")
for j in range(5):
    #####Five fold#####
    train_data, test_data = five_fold(data, j)
    train_GSM = train_data.iloc[:,0]
    test_GSM = test_data.iloc[:,0]
    #####Train Data Set#####
    train_x = train_data.iloc[:,1:-2]
    train_x = train_x.as_matrix()
    train_x = np.concatenate((train_x, np.zeros((len(train_x), 84))),axis = 1)
    train_y = train_data.iloc[:,-1].as_matrix()
    train_y = train_y.flatten()
    train_y = pd.get_dummies(train_y)

    #####Test Data Set#####
    test_x = test_data.iloc[:,1:-2]
    test_x = test_x.as_matrix()
    test_x = np.concatenate((test_x, np.zeros((len(test_x), 84))) ,axis = 1)
    test_y = test_data.iloc[:,-1].as_matrix()
    test_y = test_y.flatten()
    test_y = pd.get_dummies(test_y)
    
    cnt_train = len(train_x[1, :])
    print("This Train is No.1")
    CNN()
    ###train h를 file로
    ###test h를 file로 

