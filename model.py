import utils
import tensorflow as tf

def get_train_model():

    input=tf.placeholder(tf.float32,[None,None,utils.window_height])
    #Sparse Tensor placeholder for ctc
    targets=tf.sparse_placeholder(tf.int32)

    #seq length to be processed
    seq_len=tf.placeholder(tf.int32,[None])

    #Defining the cell
    forward_hidden_1=tf.contrib.rnn.LSTMCell(utils.num_hidden,state_is_tuple=True)
    backward_hidden_1=tf.contrib.rnn.LSTMCell(utils.num_hidden,state_is_tuple=True)

    #Setting up the network, second output is states and is not needed
    outputs,_=tf.nn.bidirectional_dynamic_rnn(forward_hidden_1,backward_hidden_1,inputs,seq_len,dtype=tf.float32)
    outputs=tf.concat(outputs,2)

    shape=tf.shape(inputs)
    batch_s,max_timesteps=shape[0],shape[1]
    #Initializing Weights with random distribition
    weights=tf.Variable(tf.truncated_normal([2*utils.num_hidden,utils.num_classes],stddev=0.1,name="W"))
    #Initializing biases with zeroes
    b=tf.zeroes(shape=[utils.num_classes],name="b")
    #Reshaping to obtain single pixel windows to apply weights over
    outputs=tf.reshape(outputs,[-1,2*utils.num_hidden])

    #Applying weights and biases to the input sequence
    logits=tf.matmul(outputs,W)+b

    #Reshaping back to the original shape
    logits=tf.reshape(logits,[batch_s,-1,utils.num_classes])

    #Swapping axis
    logits=tf.transpose(logits,(1,0,2))

    return logits,inputs,targets,seq_len
