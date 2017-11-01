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
    
#def random_five_fold(data, num, indexs):
#    test_set = data[:, indexs[:2000]]
#    train_set = data[ :, indexs[2000:]]
#    #train_set = np.concatenate((train_set,data[(num+1)*2000:] ), axis=0)
#    return train_set , test_set

def top_of_variance(per,  data_x):
    ##data_x['variance']
    ##calculate value  
    data = data_x[data_x['VAR'] > per]
    idx_list = data.index.tolist()
    ##return index 
    return idx_list

def set_train_three_layer(num,repeat, nodes, learning_rate):
    tf.reset_default_graph()
    train_a = 0
    test_a = 0
    X = tf.placeholder(tf.float32, [None, cnt_train])
    Y = tf.placeholder(tf.float32, [None, 2])

    W1 = tf.get_variable( shape= [cnt_train, nodes[0]], name='weight1' , initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([nodes[0]]), name='bias1')
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)


    W2 = tf.get_variable(shape =[nodes[0], nodes[1]], name='weight2', initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([nodes[1]]), name='bias2')
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

    W3 = tf.get_variable(shape= [nodes[1], nodes[2]], name='weight3',initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([nodes[2]]), name='bias3')
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

    W4 = tf.get_variable(shape=[nodes[2], 2], name='weight4',initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([2]), name='bias4')
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
    


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
            sess.run(train, feed_dict={X: train_x, Y: train_y})
            if step == repeat-1:
                ####Train Accuracy report####
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y  })
                print("\nTrain Accuracy: ", train_a)
            if step % 20 == 0 :
                h,c, p,train_a = sess.run([hypothesis, cost ,predicted, accuracy],feed_dict={X: train_x, Y: train_y })
                print("\nCurrent Accuracy : ", train_a , "cost : ", c , "Current Step : ", step)
                if train_a > 0.95 :
                    break
        ######Accuracy Report#####
        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y})    
        print("\nTest Accuracy: ", test_a)
    
    return train_a, test_a
        
def set_train_four_layer(num ,repeat, nodes, learning_rate):
    tf.reset_default_graph()
    train_a = 0
    test_a = 0
    X = tf.placeholder(tf.float32, [None, cnt_train])
    Y = tf.placeholder(tf.float32, [None, 2])
    
    W1 = tf.get_variable( shape= [cnt_train, nodes[0]], name='Weight1' , initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([nodes[0]]), name='Bias1')
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    
    W2 = tf.get_variable(shape =[nodes[0], nodes[1]], name='Weight2', initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([nodes[1]]), name='Bias2')
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    
    W3 = tf.get_variable(shape= [nodes[1], nodes[2]], name='Weight3',initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([nodes[2]]), name='Bias3')
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

    W4 = tf.get_variable(shape = [nodes[2], nodes[3]] , name='Weight4' , initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([nodes[3]]), name='Bias4')
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

    W5 = tf.get_variable(shape = [nodes[3], 2],name='Weight5',initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([2]), name='Bias5')
    hypothesis = tf.matmul(layer4, W5) + b5

    # cost/loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    # Accuracy computation


    predicted = tf.argmax(hypothesis,1)
    correct_prediction = tf.equal(predicted,tf.argmax(Y,1))


    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(repeat):
            sess.run(train, feed_dict={X: train_x, Y: train_y})
            if step == repeat-1:
                ####Train Accuracy report####
                h, c, train_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: train_x, Y: train_y})
                print("\nTrain Accuracy: ", train_a)
            if step % 20 == 0 : 
                h, c, p,train_a = sess.run([hypothesis, cost ,predicted, accuracy],feed_dict={X: train_x, Y: train_y })
                print("\nCurrent Accuracy : ", train_a , "Cost : ",c , "Current Step : ", step)
                if train_a > 0.95 :
                    break

        ######Accuracy Report#####
        h, c, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y})    
        print("\nTest Accuracy: ", test_a)
    
    return train_a, test_a

####Read data####
#x_filename = input("Insert X dataset directory and name  : ")
#y_filename = input("Insert Y dataset directory and name : ")
x_filename = '/home/tjahn/Data/DNN10000/DNN10000.csv'
xdata = pd.read_csv(x_filename)
ydata = np.genfromtxt('/home/tjahn/Data/DNN10000/CancerResult.csv', delimiter=",")
#conf_filename = input("Insert configure file directory and name : ")
conf_filename = '/home/tjahn/Git/Data/input/relu_test_ps.csv'
conf = pd.read_csv(conf_filename)
print(conf)
train_accs = []
test_accs = []
for i in range(len(conf)):
    repeat, layer, node , learning_rate, gene = conf.iloc[i]
    nodes = list(map(int , node.split(" ")))

    j = 0
    cnt_train = len(xdata.iloc[:,-1])    
    indexs = list(range(len(xdata.iloc[1,3:-1])))
    random.shuffle(indexs)
    variance_set = xdata.iloc[:, indexs[2000:]]
    ###############################Edit############################
    #variance_set = pd.concat([xdata.iloc[:2000*j], xdata.iloc[2000*(j+1):]])
    #print(variance_set.iloc[:,-1])
    variances = variance_set.iloc[:,-1]
    #print(variances)
    variances = variances.as_matrix()
    #print(variances)
    variances = np.sort(variances)
    #print(variances)
    idx = top_of_variance(cal_var(variances, gene) , variance_set)
    ###############################Edit############################
    
    data_x = xdata.loc[idx]
    data_x = data_x.as_matrix()
    data_x = data_x[1:, 3:-1]
    data_y = ydata[1:, 1:]    # eliminate heading, string data
    # One-Hot-Encoding
    data_y = data_y.flatten()
    data_y = pd.get_dummies(data_y)


    
    ###############################Edit############################
    test_x = data_x[:,indexs[:2000]]
    train_x = data_x[:,indexs[2000:]]
    train_x = train_x.transpose()
    test_x = test_x.transpose()
    train_y, test_y = random_five_fold(data_y,j,indexs)
    ###############################Edit############################

    #print(train_y)
    if(conf.iloc[i]['layer'] == 3):
        train_acc , test_acc = (set_train_three_layer(i,repeat, nodes, learning_rate))
    elif(conf.iloc[i]['layer']== 4):
        train_acc , test_acc = (set_train_four_layer(i,repeat, nodes, learning_rate))
    train_accs.append(train_acc)
    test_accs.append(test_acc)


train_accs = pd.DataFrame(data=train_accs , 
                          index = list(range(len(conf))) , 
                          columns = ["train_acc"])
test_accs = pd.DataFrame(data=test_accs , 
                          index = list(range(len(conf))) , 
                          columns = ["test_acc"])

accuracies = pd.concat([train_accs, test_accs], axis=1)
conf = pd.concat([conf, accuracies] , axis = 1)
conf.to_csv('/home/tjahn/Git/Data/output/'+ conf_filename[:-4] +'_result.csv' , sep= ',')


