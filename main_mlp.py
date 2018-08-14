#Author:Piento28
#Last edited:30/NOV/2017
from __future__ import division
from __future__ import print_function
import os.path


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#for visulisation embedding
from tensorflow.contrib.tensorboard.plugins import projector

import argparse
import numpy as np
# from skimage import io
# from PyInstaller.utils.hooks import collect_data_files, collect_submodules
#
# datas = collect_data_files("skimage.io._plugins")
# hiddenimports = collect_submodules('skimage.io._plugins')
# import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST')

input_dim = 784
test_num=20*20
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 2
lam = 0
n_steps = int(50000)
batch_size = 100

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial, name="W")

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial, name="B")

x = tf.placeholder("float", shape=[None, input_dim] ,name="Input_X")
test_epsilon_holder = tf.placeholder("float", shape=[test_num,latent_dim] ,name="test_epsilon")

# l2_loss = tf.constant(0.0)

with tf.name_scope("E_2hidden"):
    W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
    b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
    # l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
    hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

with tf.name_scope("E_2mean"):
    W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
    b_encoder_hidden_mu = bias_variable([latent_dim])
    # l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder
    mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

with tf.name_scope("E_2sigma"):
    W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
    b_encoder_hidden_logvar = bias_variable([latent_dim])
    # l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
    logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='Epsilon')

with tf.name_scope("Sampling"):
# Sample latent variable
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(std_encoder, epsilon)

with tf.name_scope("D_2hidden"):
    W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
    b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
    # l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
    hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)
    test_hidden_decoder = tf.nn.relu(tf.matmul(test_epsilon_holder, W_decoder_z_hidden) + b_decoder_z_hidden)

with tf.name_scope("D_2output"):
    W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
    b_decoder_hidden_reconstruction = bias_variable([input_dim])
    # l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)
    
    x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
    test_x_hat = tf.matmul(test_hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
    
with tf.name_scope("KLD_loss"):
    KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

with tf.name_scope("BCE_loss"):
    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)

with tf.name_scope("Regularised_Loss"):
    loss = tf.reduce_mean(BCE + KLD)
    # loss = tf.reduce_mean(BCE)

    regularized_loss = loss #+ lam * l2_loss

loss_summ = tf.summary.scalar("lowerbound", loss)
summary_image_input = tf.summary.image("input",tf.reshape(x,[-1,28,28,1]),10)
summary_image_output = tf.summary.image("output",tf.reshape(x_hat,[-1,28,28,1]),10)
# summary_image_latent = tf.summary.image("latent",tf.reshape(z,))
with tf.name_scope("Train"):
    train_step = tf.train.AdamOptimizer(0.001).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")
    
    parser.add_argument('--trainOtest', action='store', dest='train_test', default='train', choices=['train', 'visua'])
    
    args = parser.parse_args()

if args.train_test == "train":
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('experiment', graph=sess.graph)
        # ################################################################################
        # # Create randomly initialized embedding weights which will be trained.
        # N = 10000 # Number of items (vocab size).
        # D = 200 # Dimensionality of the embedding.
        # embedding_var = tf.Variable(tf.random_normal([784]), name='test_embedding')
        #
        # # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        # config = projector.ProjectorConfig()
        #
        # # You can add multiple embeddings. Here we add only one.
        # embedding = config.embeddings.add()
        # embedding.tensor_name = embedding_var.name
        # # Link this tensor to its metadata file (e.g. labels).
        # # embedding.metadata_path = os.path.join('experiment' + '/projector/metadata.tsv')
        # embedding.sprite.image_path = os.path.join('experiment', 'mnist_10k_sprite.png')
        # # Specify the width and height of a single thumbnail.
        # embedding.sprite.single_image_dim.extend([28, 28])
        # # Use the same LOG_DIR where you stored your checkpoint.
        # # summary_writer = tf.summary.FileWriter('experiment')
        #
        # # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # # read this file during startup.
        # projector.visualize_embeddings(summary_writer, config)
        # ################################################################################
        #
        # ################################################################################
        if os.path.isfile("experiment/model.ckpt"):
            print("Restoring saved parameters")
            saver.restore(sess, "experiment/model.ckpt")
        else:
            print("Initializing parameters")
            sess.run(tf.global_variables_initializer())
        
        min_loss=999999.9
        
        for step in range(1, n_steps):
            batch = mnist.train.next_batch(batch_size)
            feed_dict = {x: batch[0]}
            _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            if step % 50 == 0:
                if (cur_loss<=min_loss):
                    save_path = saver.save(sess, "experiment/model.ckpt")
                    min_loss=cur_loss
                print("Step {0} | Loss: {1}".format(step, cur_loss))
else:
    with tf.Session() as sess:
        # summary_writer = tf.summary.FileWriter('experiment', graph=sess.graph)
        if os.path.isfile("experiment/model.ckpt.meta"):
            print("Restoring saved parameters\n")
            print("==================================================\n")
            saver.restore(sess, "experiment/model.ckpt")
        else:
            print("Error:There is no such model.ckpt in logdir")
            exit()
        test_epsilon=np.zeros([test_num,latent_dim], dtype=np.float32)
        for i in range(20):
            for j in range(20):
                test_epsilon[i*20+j][0]=-5+i/2
                test_epsilon[i*20+j][1]=-5+j/2

        test_out=sess.run([test_x_hat],feed_dict={test_epsilon_holder:test_epsilon})
        print(np.shape(test_out))
        test_out=np.reshape(test_out,[-1,28,28])
        print(np.shape(test_out))
        np.save("./experiment/via.npy",test_out)
