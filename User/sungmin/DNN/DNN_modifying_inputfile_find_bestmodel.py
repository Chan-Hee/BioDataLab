##################Import Modules#######################
import random
import numpy as np
import math
import pandas as pd
import tensorflow as tf
from scipy import stats


tf.set_random_seed(777)


##################Define Functions#####################
def five_fold_name(data,i):
    test_names = data[data['index']==i+1]


def five_fold(data, i):
    test_data = data[data['index']==i+1]
    train_data = data[(data['index']<i+1) | (data['index']>i+1)]
    print(len(test_data), len(train_data))

    return train_data , test_data

def set_train_three_layer(nodes, learning_rate, j, gene_off):
    batch_size = 1000
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32)

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
    cost_summ = tf.summary.scalar(str(int(gene_off))+"_"+str(nodes)+"_cost",cost)



    # Accuracy computation
    # True if hypothesis>0.5 else False

    predicted = tf.argmax(hypothesis,1)
    correct_prediction = tf.equal(predicted,tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    accuracy_summ = tf.summary.scalar(str(int(gene_off))+"_"+str(nodes)+"_accuracy",accuracy)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/home/tjahn/tf_save_data/sungmin/tensorboard/node_layer_result/gene_off_"+str(int(gene_off)) +"/" + str(nodes) +"_" + str(j+1))
        writer.add_graph(sess.graph)

        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        stop_switch = True
        step=0
        AccuracyList=[]
        
        max_step = 0
        max_Accuracy = 0

        while(stop_switch):
            total_num = int(len(train_x)/batch_size)
            for i in range(total_num):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                sess.run(train , feed_dict={X: batch_x, Y: batch_y , keep_prob : 1})
                summary,_=sess.run([merged_summary,train], feed_dict={X: batch_x, Y: batch_y , keep_prob : 1})
                writer.add_summary(summary, global_step =step)

            train_h,c, train_p,train_a = sess.run([hypothesis, cost ,predicted, accuracy],feed_dict={X: train_x, Y: train_y, keep_prob :1})
            cal_h,c, cal_p,cal_a = sess.run([hypothesis, cost ,predicted, accuracy],feed_dict={X: cal_x, Y: cal_y, keep_prob :1})
            step+=1
            print("\nTraining Accuracy : ", train_a , "Calibration Accuracy : ", cal_a,"Step :", step)
            if len(AccuracyList) == 20:
                AccuracyList.pop(0)
                AccuracyList.append(cal_a)
                beforeAccuracy = AccuracyList[:int(len(AccuracyList)/2)]
                afterAccuracy = AccuracyList[int(len(AccuracyList)/2):]
                tTestResult = stats.ttest_rel(beforeAccuracy,afterAccuracy)
                print("P-Value: ",tTestResult.pvalue,"\n",beforeAccuracy,"\n",afterAccuracy)

#save path
                if(max(AccuracyList)> max_Accuracy):
                    max_step = step
                    max_Accuracy = max(AccuracyList)
                    save_path = saver.save(sess, '/home/tjahn/tf_save_data/sungmin/save_path/saved')
                    print("Save path: ",save_path,"\nMax_step: ",max_step,"\nMax_Accuracy: ",max_Accuracy )
                 
                   # print("W1, W2, W3, W4: ", W1.eval(),W2.eval(),W3.eval(), W4.eval())

                    # print("W1, W2, W3, W4: " W1.eval() + " "+ W2.eval()+ " " +W3.eval() + " " + W4.eval())
                if max(AccuracyList)-min(AccuracyList)< 0.01 and min(AccuracyList)>0.94 and max(beforeAccuracy) >= max(afterAccuracy) :
                    stop_switch = False
                    print("Learning Finished!! P-Value: ",tTestResult.pvalue,"\n",beforeAccuracy,"\n",afterAccuracy)
                   # print("W1, W2, W3, W4: %s  %s  %s  %s \n", W1.eval(),W2.eval(),W3.eval(), W4.eval())

                   # print("W1, W2, W3, W4: "+ W1.eval() + " "+ W2.eval()+ " " +W3.eval() + " " + W4.eval())
           

            else:
                AccuracyList.append(cal_a)



        w1_matrix=W1.eval()

        weighted_sum = w1_matrix.sum(axis=1)
        weighted_max = w1_matrix.max(axis=1)
        gene_names = list(data)[1:-2]

        print("Save path: ",save_path)
        saver.restore(sess,save_path)
        print("Max_step: ",max_step,"Max_accuracy: ", max_Accuracy)
       # print("W1, W2, W3, W4: "+ W1.eval() + " "+ W2.eval()+ " " +W3.eval() + " " + W4.eval())
      #  print("W1, W2, W3, W4: %s  %s  %s  %s \n", W1.eval(),W2.eval(),W3.eval(), W4.eval())

        test_h, test_p, test_a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: test_x, Y: test_y, keep_prob :1.0})
        print("\nTest Accuracy: ", test_a)

        weighted_sum_result = [str(int(gene_off)),train_a,cal_a,test_a]
    #    weighted_sum_result = [gene_off,train_a,cal_a,test_a,max_step,max_Accuracy]
         
    return train_p ,train_h, test_p,test_h,weighted_sum_result

##################READ DATA############################
repeat, learning_rate = 1000, 0.002

output_directory = "/home/tjahn/tf_save_data/sungmin/result_to_find_best_model/"
conf_directory = "/home/tjahn/Git2/User/sungmin/DNN/input/"
conf_filename = "input_many_70.csv"
conf = pd.read_csv(conf_directory + conf_filename)


###
for i in range(len(conf)):
    layer, node, gene_off = conf.iloc[i]
    nodes = list(map(int, node.split(" ")))
   # gene_off = conf.iloc[i]

####sm
    datafilename = "/home/tjahn/Data/FinalData"+str(int(gene_off))+"off_GSM_gene_index_result.csv"
    data = pd.read_csv(datafilename)
####sm
    Gene_Elimination = []
    Training_Accuracy=[]
    Calibration_Accuracy=[]
    Test_Accuracy=[]
  #  Max_step = []
  #  Max_Accuracy = []
    for j in range(5):
    #####Five fold#####
        train_data, test_data = five_fold(data, j)
        cal_data = test_data[:int(len(test_data)/2)]
        test_data = test_data[int(len(test_data)/2):]
        train_GSM = train_data.iloc[:,0].tolist()
        test_GSM = test_data.iloc[:,0].tolist()
        cal_GSM = cal_data.iloc[:,0].tolist()

    #####Train Data Set#####
        train_x = train_data.iloc[:,1:-2]
        train_x = train_x.as_matrix()
        train_y = train_data.iloc[:,-1].as_matrix()
        train_y = train_y.flatten()
        train_y = pd.get_dummies(train_y)
        cnt_train = len(train_x[1, :])
    #    nodes = [int(cnt_train),int(cnt_train/2),int(cnt_train/4)]

    #####Test Data Set#####
        test_x = test_data.iloc[:,1:-2]
        test_x = test_x.as_matrix()
        test_y = test_data.iloc[:,-1].as_matrix()
        test_y = test_y.flatten()
        test_y = pd.get_dummies(test_y)

    ####Cal Data Set#####
        cal_x = cal_data.iloc[:,1:-2]
        cal_x = cal_x.as_matrix()
        cal_y = cal_data.iloc[:,-1].as_matrix()
        cal_y = cal_y.flatten()
        cal_y = pd.get_dummies(cal_y)


        train_p, train_h , test_p ,test_h, weighted_sum_result  = (set_train_three_layer(nodes, learning_rate, j, gene_off))
        train_p = pd.DataFrame(train_p, index = train_GSM )
        train_h = pd.DataFrame(train_h , index = train_GSM)
        test_p = pd.DataFrame(test_p , index = test_GSM)
        test_h = pd.DataFrame(test_h, index = test_GSM)
        train_y = pd.DataFrame(train_data.iloc[:,-1].as_matrix() , train_GSM)
        test_y = pd.DataFrame(test_data.iloc[:,-1].as_matrix(), test_GSM)

        train_result = pd.concat([train_y, train_p], axis = 1 )
        train_result = pd.concat([train_result, train_h], axis =1)

        test_result = pd.concat([test_y ,test_p] , axis =1 )
        test_result = pd.concat([test_result, test_h], axis=1)

        train_result.columns = ['result','prediction','prob0', 'prob1' ]
        test_result.columns = ['result', 'prediction', 'prob0', 'prob1']

        Gene_Elimination.append(weighted_sum_result[0])
        Training_Accuracy.append(weighted_sum_result[1])
        Calibration_Accuracy.append(weighted_sum_result[2])
        Test_Accuracy.append(weighted_sum_result[3])
      #  Max_step.append(weighted_sum_result[4])
      #  Max_Accuracy.append(weighted_sum_result[5])
## Accuracy Data 생성 ##
   # Accuracy_Dataframe = pd.DataFrame({"Gene_Elimination":Gene_Elimination,"Training_Accuracy":Training_Accuracy,"Calibration_Accuracy":Calibration_Accuracy,"Test_Accuracy":Test_Accuracy,"Max_step":Max_step,"Max_Accuracy":Max_Accuracy})

    Accuracy_Dataframe = pd.DataFrame({"Gene_Elimination":Gene_Elimination,"Training_Accuracy":Training_Accuracy,"Calibration_Accuracy":Calibration_Accuracy,"Test_Accuracy":Test_Accuracy})


#### 파일 생성 ####

    #result_train_filename = "result_file_train"+ gene_off + str(j) +".csv"
    #train_result.to_csv(output_directory+result_train_filename , sep= ',')
    #result_test_filename = "result_file_test" + gene_off +str(j) +".csv"
    #test_result.to_csv(output_directory+result_test_filename , sep= ',')

    ###train h를 file로
    ###test h를 file로

    Accuracy_Dataframe_filename="result_weigthed_sum_gene_"+str(int(gene_off))+"percent_off_"+str(nodes)+".csv"
    Accuracy_Dataframe.to_csv(output_directory+Accuracy_Dataframe_filename,sep=",")
