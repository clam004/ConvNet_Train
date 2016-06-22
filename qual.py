from functions.conv1fxn import *
from functions.functions_image_training import *
import tensorflow as tf
import timeit
import numpy as np
import random
import h5py
from scipy.misc import imresize

with h5py.File('data/qualdataZ.h5', 'a') as qd:
    num_images_labels = len(list(qd.keys()))
    label = np.array(qd.get("label1"))

############## INPUTS: including bacth size, list of tuples and image dictionary###########################
batch_size = 21
epochs = 2
single_input_shape = (450,450,3)
validation_proportion = 0.025
num_classes = label.shape[0]

target_height,target_width,target_depth = single_input_shape
num_examples = int(num_images_labels / 2)
example_index = list(range(num_examples))
random.shuffle(example_index)
start_of_training = int(validation_proportion*num_examples)
validation_index = example_index[:start_of_training]
training_index = example_index[start_of_training:]
num_val = len(validation_index)
num_train = len(training_index)
num_steps = int(num_train / batch_size)

val_label = np.zeros((num_val,num_classes))
val_image = np.zeros((num_val,target_height,target_width,target_depth))

for i in range(num_val):
    with h5py.File('data/qualdataZ.h5', 'a') as qd:
            val_raw_image = np.array(qd.get("image"+str(validation_index[i])))
            val_label[i] = np.array(qd.get("label"+str(validation_index[i])))
            val_raw_image = val_raw_image[:,70:530,:] # consider removing me
            val_image[i] = imresize(val_raw_image,(450,450,3)) /255

################ TENSOR FLOW #########################################
### TF variables #####################################################
tensor_image = tf.placeholder("uint8", [None, None, 3])

############ Manipulations to tensors in TF variables ########################
#### Resizing tesors #################################################################
resized_image = tf.image.resize_images(tensor_image, target_height, target_width)

######## Manipulations to resized tensors in Resizing tensors ####################
lrflip_resized = tf.image.flip_left_right(resized_image)
udflip_resized = tf.image.flip_up_down(resized_image)

####### BUILD NEURAL NET ARCHITECTURE ########################################
x = tf.placeholder(tf.float32, shape=[None, target_height,target_width,target_depth])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

W_conv1 = weight_variable([10, 10, 3, 48])
b_conv1 = bias_variable([48])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
print('h_conv1 shape', h_conv1.get_shape().as_list())

h_pool1 = max_pool_2x2(h_conv1)
print('h_pool1 shape', h_pool1.get_shape().as_list())

W_conv2 = weight_variable([5, 5, 48, 40])
b_conv2 = bias_variable([40])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print('h_conv2 shape', h_conv2.get_shape().as_list())

W_conv3 = weight_variable([5, 5, 40, 32])
b_conv3 = bias_variable([32])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

h_conv3_shape =  h_conv3.get_shape().as_list()
print('h_conv3_shape', h_conv3_shape)

W_fc1 = weight_variable([h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3], 256])
b_fc1 = bias_variable([256])

# reshape x to 4d tensor[-1,width,height,depth, channels]
h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]])
print('h_conv3_flat shape', h_conv3_flat.get_shape().as_list())

h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([256, 128])
b_fc2 = bias_variable([128])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
print('h_fc2  shape', h_fc2 .get_shape().as_list())

# probability that a neuron's output is kept during dropout
keep_prob = tf.placeholder(tf.float32)

#dropout layer
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
print('h_fc1_drop shape', h_fc2_drop.get_shape().as_list())

W_fc3 = weight_variable([128, num_classes])
b_fc3 = bias_variable([num_classes])

#################### SOFTMAX layer ###############################
h_fc3_logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

y_soft=tf.nn.softmax(h_fc3_logits)
print('y_soft shape', y_soft.get_shape().as_list())

###################### REGULARIZATION ##############################
L2_reg_const = 1e-6

L2_reg = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3)
+ tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3))

#DEFINE COST FXN as cross entropy loss
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_soft,1e-10,1.0)))  + L2_reg_const*L2_reg

#DEFINE OPTIMIZER Instance
#train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.99,staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
#evaluate accuracy as boolean
correct_prediction = tf.equal(tf.argmax(y_soft,1), tf.argmax(y_,1))
#evaluate accuracy as binary average
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


######## initialization class for session running ############################
model_and_variables = tf.initialize_all_variables()
print('session initialized variables')

start_time = timeit.default_timer()

################## START SESSION ###############################################
with tf.Session() as session:

    saver = tf.train.Saver()
    saved_model = "models/qualZ2.ckpt"

    if os.path.isfile(saved_model):
        print('Found', saved_model)
        saver.restore(session, saved_model)
        print(saved_model, "Model restored.")
    else:
        print('No saved Model')

    best_val_acc = np.float32(1/num_classes)

############## INITIALIZE VARIABLES WITHIN THIS SESSION #############################
    session.run(model_and_variables)
########## TRAINING LOOP #####################################################

########## NUMBER OF EPOCHS ##################################################
    for epoch in range(epochs):
        print('epoch', epoch, 'steps:', num_steps)
    ########## SHUFFLE TRAINING DATA BETWEEN EPOCHS #############################
        random.shuffle(training_index)
    ########## ONE EPOCH #####################################################
        for step in range(num_steps):
            offset = (step * batch_size) % (num_train)
            start = offset
            end = offset + batch_size
            if num_train - end < batch_size:
                end += (num_train % batch_size) # last end is extended to end of training set

            effective_batch_size = end - start
    ######## INITIALIZE TRAINING BATCH ########################################################
            batch_label = np.zeros((effective_batch_size,num_classes))
            batch_image = np.zeros((effective_batch_size,target_height,target_width,target_depth))
            j = 0
            for i in range(start,end):
    ####### USE 'start' and 'end' to build one BATCH #####################################
    ####### HERE ################################################################
                with h5py.File('data/qualdataZ.h5', 'a') as qd:
                        batch_raw_image = np.array(qd.get("image"+str(training_index[i])))
                        batch_label[j] = np.array(qd.get("label"+str(training_index[i])))
                        batch_raw_image = batch_raw_image[:,70:530,:] # consider removing me
                        batch_image[j] = imresize(batch_raw_image,(450,450,3)) /255

                j+=1
########### FEED TRAINING BATCH TO NETWORK ##########################################
########### HERE ####################################################################

            train_batch,Loss = session.run([train_step,cross_entropy],
                               feed_dict={x: batch_image, y_: batch_label, keep_prob: 1.0})

            print('finished step ', step, 'learning rate' ,learning_rate.eval(), 'Loss:',Loss)
            print("max W vales: %g %g %g %g %g %g"%(tf.reduce_max(tf.abs(W_conv1)).eval(),
                                              tf.reduce_max(tf.abs(W_conv2)).eval(),tf.reduce_max(tf.abs(W_conv3)).eval(),
                                              tf.reduce_max(tf.abs(W_fc1)).eval(),tf.reduce_max(tf.abs(W_fc2)).eval(),
                                              tf.reduce_max(tf.abs(W_fc3)).eval()))

            print("max b vales: %g %g %g %g %g"%(tf.reduce_max(tf.abs(b_conv1)).eval(),tf.reduce_max(tf.abs(b_conv2)).eval(),
                                              tf.reduce_max(tf.abs(b_fc1)).eval(),tf.reduce_max(tf.abs(b_fc2)).eval(),
                                              tf.reduce_max(tf.abs(b_fc3)).eval()))

            print("min W vales: %g %g %g %g %g %g"%(tf.reduce_min(tf.abs(W_conv1)).eval(),
                                              tf.reduce_min(tf.abs(W_conv2)).eval(),tf.reduce_min(tf.abs(W_conv3)).eval(),
                                              tf.reduce_min(tf.abs(W_fc1)).eval(),tf.reduce_min(tf.abs(W_fc2)).eval(),
                                              tf.reduce_min(tf.abs(W_fc3)).eval()))

            batch_acc = session.run(accuracy, feed_dict={x: batch_image, y_: batch_label, keep_prob: 1.0})
            print("batch accuracy: %g"% batch_acc)
            valid_acc = session.run(accuracy, feed_dict={x: val_image, y_: val_label,  keep_prob: 1.0})
            print("Validation Accuracy: %g"% valid_acc)
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                print('NEW best_val_acc')
                save_path = saver.save(session, saved_model)
                print("Model saved in file: %s" % save_path)
#######################################################################################

end_time = timeit.default_timer()
print('DONE, ran for %.2fm' % ((end_time - start_time) / 60.))
