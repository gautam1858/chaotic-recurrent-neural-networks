import tensorflow as tf
import numpy as np

class ChaoticRNN(object):
    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length):
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=n_steps
        n_layers=3
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)
        print(x)
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            print(tf.get_variable_scope().name)
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=dropout)
            lstm_fw_cell_m=tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            print(tf.get_variable_scope().name)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=dropout)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2

    def __init__(
      self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):
      self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
      self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
      self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      l2_loss = tf.constant(0.0, name="l2_loss")

      with tf.name_scope("embedding"):
          self.W = tf.Variable(
              tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
              trainable=True,name="W")
          self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
          self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
      with tf.name_scope("output"):
        self.out1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length)
        self.out2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length)
        self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.out1,self.out2)),1,keep_dims=True))
        self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1], name="distance")
      with tf.name_scope("loss"):
          self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
      with tf.name_scope("accuracy"):
          correct_predictions = tf.equal(self.distance, self.input_y)
          self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
