import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

tf.set_random_seed(42)
np.random.seed(42)

with open(r'mnist_short.pickle', 'rb') as fb:
    data = pickle.load(fb)
    
with open(r'mnist_short_test.pickle', 'rb') as fb:
    data_test = pickle.load(fb)
    
x_train = data['x']
x_test = data_test['x']
    
y_train = data['y'].flatten()
y_test = data_test['y'].flatten()

training_label_zeros = np.zeros((x_train.shape[0], len(np.unique(y_train))))
training_label_zeros[np.arange(x_train.shape[0]), y_train.flatten().astype(int)] = 1
y_train = training_label_zeros

test_label_zeros = np.zeros((x_test.shape[0], len(np.unique(y_test))))
test_label_zeros[np.arange(x_test.shape[0]), y_test.astype(int)] = 1
y_test = test_label_zeros

#Normilizing the input
m = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0) + 1e-10
x_train = (x_train - m) / std
x_test = (data_test['x'] - m) / std

#Initializing the parameters
def init_param(shape):
    
    param = tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype='float', seed=None, name=None))

    return param

#Forward pass
def nn1(x, w1, b1, w2, b2, w3, b3, w4, b4):
    a1 = tf.add(tf.matmul(x, w1,  transpose_b=True), b1)
    z1 = tf.nn.sigmoid(a1)
    a2 = tf.add(tf.matmul(z1, w2,  transpose_b=True), b2)
    z2 = tf.nn.sigmoid(a2)
    a3 = tf.add(tf.matmul(z2, w3,  transpose_b=True), b3)
    z3 = tf.nn.sigmoid(a3)
    y_p = tf.add(tf.matmul(z3, w4, transpose_b=True), b4)
    return tf.nn.sigmoid(y_p)

if __name__ == '__main__':
    
    n, d = x_train.shape
    o = y_train.shape[1]
    lr = 0.1
    mb_size = 10
    h_nodes = 30
    
    X = tf.placeholder('float', shape=(None, d))
    y = tf.placeholder('float', shape=(None, o))
    
    w1 = init_param((h_nodes, d))
    b1 = init_param((h_nodes, ))
    
    w2 = init_param((h_nodes, h_nodes))
    b2 = init_param((h_nodes, ))
    
    w3 = init_param((h_nodes, h_nodes))
    b3 = init_param((h_nodes, ))
    
    w4 = init_param((o, h_nodes))
    b4 = init_param((o, ))
    
    pred = nn1(X, w1, b1, w2, b2, w3, b3, w4, b4)
    yhat = tf.argmax(pred, axis=1)
    
    lam = 5
    lam = tf.constant(lam / n, dtype='float')
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + lam*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4))#lam*(tf.reduce_sum(tf.pow(w1, 2)) + tf.reduce_sum(tf.pow(w2, 2)) + tf.reduce_sum(tf.pow(w3, 2)) + tf.reduce_sum(tf.pow(w4, 2)))
    
    update = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    
    w10 = sess.run(w1)
    w20 = sess.run(w2)
    w30 = sess.run(w3)
    w40 = sess.run(w4)

    train_loss_l = []
    test_loss_l = []
    train_accuracy_l = []
    test_accuracy_l = []
    sp1 = []
    sp2 = []
    sp3 = []
    sp4 = []
    start_time = time.time()
    
    for epoch in range(30):
        for i in range(int(n / 10)):
            
            idx = i * mb_size
            sess.run(update, feed_dict={
                X: x_train[idx: idx+mb_size, :],
                y: y_train[idx:idx+mb_size, :] 
            })
            
            w11 = sess.run(w1)
            w21 = sess.run(w2)
            w31 = sess.run(w3)
            w41 = sess.run(w4)
            
            sp1.append(np.mean(np.abs((w11-w10)/w10)))
            
            sp2.append(np.mean(np.abs((w21-w20)/w20)))

            sp3.append(np.mean(np.abs((w31-w30)/w30)))

            sp4.append(np.mean(np.abs((w41-w40)/w40)))

            w10 = sess.run(w1)
            w20 = sess.run(w2)
            w30 = sess.run(w3)
            w40 = sess.run(w4)

            train_loss = sess.run(loss, feed_dict={X: x_train, y: y_train})
            
            test_loss = sess.run(loss, feed_dict={X: x_test, y: y_test})
            
            train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(yhat, feed_dict={X: x_train}))
            
            test_accuracy = np.mean(np.argmax(y_test, axis=1) == sess.run(yhat, feed_dict={X: x_test}))
            
            train_loss_l.append(train_loss)
            test_loss_l.append(test_loss)
            train_accuracy_l.append(train_accuracy)
            test_accuracy_l.append(test_accuracy)
            
        print(f'Epoch = {epoch + 1}, train accuracy = {train_accuracy * 100:{1}.{3}}%, ',
              f'test accuracy = {test_accuracy * 100:{1}.{3}}%, \n',
              f'train loss = {train_loss:{1}.{3}}, test loss = {test_loss:{1}.{3}}')
        
    end_time = time.time()
    
    print("Time usage " + str(int(end_time - start_time)) + " seconds")
    
    #plt.subplot(121)
    plt.plot(train_accuracy_l, label='Train')
    plt.plot(test_accuracy_l, label='Test')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()

    #plt.subplot(122)
    plt.plot(train_loss_l, label='Train')
    plt.plot(test_loss_l, label='Test')
    plt.legend()
    plt.title('Loss')
    
    plt.show()
     
    print('Learning speed of the hidden layer 1: ', np.mean(sp1))
    print('Learning speed of the hidden layer 2: ', np.mean(sp2))
    print('Learning speed of the hidden layer 3: ', np.mean(sp3))
    print('Learning speed of the hidden layer 3: ', np.mean(sp4))

    plt.plot(np.log(sp1), label='Learning Speed Layer 1')
    plt.plot(np.log(sp2), label='Learning Speed Layer 2')
    plt.plot(np.log(sp3), label='Learning Speed Layer 3')
    plt.plot(np.log(sp4), label='Learning Speed Layer 4')
    plt.legend()
    plt.title('Learning Speed')

    plt.show()
    
    sess.close()