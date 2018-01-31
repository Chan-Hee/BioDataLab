from Models import Models
import numpy as np
import tensorflow as tf


class DNN(Models):
    #Constructor


    def __init__(self, nodes , activationFunction = "ReLU"):
        """
        :param
        1. nodes: nodes list of Deep neural network.
        2. activationFunction : choice of activation function , default is ReLU
        """
        super().__init__()
        self.learning_rate = 0.02
        self.gene_off = 0
        self.nodes = nodes
        self.layer = len(self.nodes)
        self.five_fold_count = 1
        self.activationFunction = activationFunction

    def get_name(self):
        return "Deep neural network"

    def fit(self, x, y , cal_x, cal_y):
        tensorboard_directory = "/Users/taewan/Desktop/"
        save_path_directory = "/Users/taewan/Desktop/"
        """
        :param
        1. x is input of train data set
        2. y is result of train data set
        """

        np.random.seed(777)
        self.keep_prob = tf.placeholder(tf.float32)

        self.weights = []
        self.bias = []
        self.hidden_layers = []
        self.batch_size = 1000

        self.X = tf.placeholder(tf.float32, [None, len(x[0])])
        self.Y = tf.placeholder(tf.float32, [None, 2])

        for i in range(len(self.nodes)) :
            if i == 0 :
                self.weights.append(tf.get_variable(shape=[len(x[0]), self.nodes[i]], name='weight' + str(i),
                                         initializer=tf.contrib.layers.xavier_initializer()))
                self.bias.append(tf.Variable(tf.random_normal([self.nodes[i]]), name='bias' + str(i)))
                self.hidden_layers.append(tf.nn.relu(tf.matmul(self.X, self.weights[i]) + self.bias[i]))
                self.hidden_layers[i] = tf.nn.dropout(self.hidden_layers[i], keep_prob=self.keep_prob)
            else :
                self.weights.append(tf.get_variable(shape=[self.nodes[i - 1], self.nodes[i]], name='weight' + str(i),
                                         initializer=tf.contrib.layers.xavier_initializer()))
                self.bias.append(tf.Variable(tf.random_normal([self.nodes[i]]), name='bias' + str(i)))
                self.hidden_layers.append(tf.nn.relu(tf.matmul(self.hidden_layers[i - 1], self.weights[i]) + self.bias[i]))
                self.hidden_layers[i] = tf.nn.dropout(self.hidden_layers[i], keep_prob=self.keep_prob)

        i = len(self.nodes)-1
        self.weights.append(tf.get_variable(shape=[self.nodes[i], 2], name='weight' + str(i + 1),
                             initializer=tf.contrib.layers.xavier_initializer()))
        self.bias.append(tf.Variable(tf.random_normal([2]), name='bias' + str(i + 1)))
        self.hypothesis = tf.matmul(self.hidden_layers[i], self.weights[i + 1]) + self.bias[i + 1]

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
        self.train = tf.train.AdamOptimizer(learning_rate= self.learning_rate ).minimize(self.cost)
        cost_summ = tf.summary.scalar(str(int(self.gene_off)) + "_" + str(self.layer) + "_" + str(self.nodes) + "_cost", self.cost)

        # Accuracy computation
        # True if hypothesis>0.5 else False

        self.predicted = tf.argmax(self.hypothesis, 1)
        self.correct_prediction = tf.equal(self.predicted, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype=tf.float32))
        accuracy_summ = tf.summary.scalar(str(int(self.gene_off)) + "_" + str(self.layer) + "_" + str(self.nodes) + "_accuracy", self.accuracy)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                tensorboard_directory + "gene_off_" + str(int(self.gene_off)) + "/" + str(self.layer) + "/" + str(self.nodes) + "_" + str(
                    self.five_fold_count + 1))
            writer.add_graph(sess.graph)

            # Initialize TensorFlow variables
            sess.run(tf.global_variables_initializer())
            stop_switch = True
            step = 0
            AccuracyList = []

            max_step = 0
            max_Accuracy = 0

            while (stop_switch):
                total_num = int(len(x) / self.batch_size)
                for i in range(total_num):
                    batch_x = x[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_y = y[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(self.train, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1})
                    summary, _ = sess.run([merged_summary, self.train], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1})
                    writer.add_summary(summary, global_step=step)

                train_h, c, train_p, train_a = sess.run([self.hypothesis, self.cost, self.predicted, self.accuracy],
                                                        feed_dict={self.X: x, self.Y: y, self.keep_prob: 1})
                cal_h, c, cal_p, cal_a = sess.run([self.hypothesis, self.cost, self.predicted, self.accuracy],
                                                  feed_dict={self.X: cal_x, self.Y: cal_y, self.keep_prob: 1})
                step += 1
                print("\nTraining Accuracy : ", train_a, "Calibration Accuracy : ", cal_a, "Step :", step)
                if len(AccuracyList) == 20:
                    AccuracyList.pop(0)
                    AccuracyList.append(cal_a)
                    beforeAccuracy = AccuracyList[:int(len(AccuracyList) / 2)]
                    afterAccuracy = AccuracyList[int(len(AccuracyList) / 2):]
                    # tTestResult = stats.ttest_rel(beforeAccuracy,afterAccuracy)
                    # print("P-Value: ",tTestResult.pvalue,"\n",beforeAccuracy,"\n",afterAccuracy)

                    # save path
                    if (max(AccuracyList) > max_Accuracy):
                        max_step = step
                        max_Accuracy = max(AccuracyList)
                        save_path = saver.save(sess, save_path_directory)
                        print("Save path: ", save_path, "\nMax_step: ", max_step, "\nMax_Accuracy: ", max_Accuracy)

                        # Early stopping
                    if max(AccuracyList) - min(AccuracyList) < 0.01 and min(AccuracyList) > 0.94 and max(
                            beforeAccuracy) >= max(afterAccuracy):
                        stop_switch = False
                        print("Learning Finished!! \n", beforeAccuracy, "\n", afterAccuracy)
                    stop_switch = False
                else:
                    AccuracyList.append(cal_a)

            # w1_matrix=W1.eval()

            # weighted_sum = w1_matrix.sum(axis=1)
            # weighted_max = w1_matrix.max(axis=1)
            # gene_names = list(data)[1:-2]

            # saver.restore(sess, save_path)
            print("Max_step: ", max_step, "Max_accuracy: ", max_Accuracy)
            # print("W1, W2, W3, W4: "+ W1.eval() + " "+ W2.eval()+ " " +W3.eval() + " " + W4.eval())
            #  print("W1, W2, W3, W4: %s  %s  %s  %s \n", W1.eval(),W2.eval(),W3.eval(), W4.eval())

            # weighted_sum_result = [str(int(gene_off)), train_a, cal_a, test_a]
            #    weighted_sum_result = [gene_off,train_a,cal_a,test_a,max_step,max_Accuracy]


    def predict(self, x ):
        self.X = x
