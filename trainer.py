import utils
import editdistance
import time
from model import get_train_model
from utils import decode_sparse_tensor
import tensorflow as tf


def report_accuracy(decoded_list,test_targets,test_names):
    original_list=decode_sparse_tensor(test_targets)
    detected_list=decode_sparse_tensor(decoded_list)
    names_list=test_names.tolist()

    total_edits=0
    total_length=0

    if len(original_list)!= len(detected_list):
        print("The length of output list: ",len(original_list)," did not match the ground truth: ",len(original_list)," skipping this batch")
        return
    print("Ground Truth Length: ",len(original_list)," Output Length: ",len(detected_list))
    for idx,current_gt in enumerate(original_list):
        current_output=detected_list[idx]
        edits=editdistance.eval(current_gt,current_output)
        length=len(current_gt)
        edit_accuracy=(length-edits)/length
        print("Output:  ",current_output)
        print("GT:      ",current_gt)
        print("Edits:   ",edits)
        print("Accuracy :",edit_accuracy)
        total_edits+=edit_accuracy
        total_length+=length
    total_accuracy=(total_length-total_edits)/total_length
    print("Set Accuracy: ",total_accuracy)
    return(total_accuracy,total_length,total_edits)

'''
def train():
    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(utils.initial_learning_rate,global_step,utils.decay_steps,
                                            utils.learning_rate_decay_factor,staircase=True)
    logits,inputs,targets,seq_len=get_train_model()
    loss=tf.nn.ctc_loss(logits,targets,seq_len)
    cost=tf.reduce_mean(loss)

    optimzer=tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=utils.momentum).mimnimize(cost,global_step=global_step)
    decoded,log_prob=tf.nn.ctc_beam_search_decoder(logits,seq_len,merge_repeated=False)
    accuracy=tf.reduce_mean(tf.editdistance(tf.cast(decoded[0],tf.int32),targets))
    return optimizer, decoded,accuracy
'''


#Defining the training graph
train=tf.Graph()
with train.as_default():
    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(utils.initial_learning_rate,global_step,utils.decay_steps,
                                            utils.learning_rate_decay_factor,staircase=True)
    logits,inputs,targets,seq_len=get_train_model()
    #print(type(logits)," ",type(targets))
    loss=tf.nn.ctc_loss(targets,logits,seq_len)
    cost=tf.reduce_mean(loss)

    optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=utils.momentum).minimize(cost,global_step=global_step)
    decoded,log_prob=tf.nn.ctc_beam_search_decoder(logits,seq_len,merge_repeated=False)
    accuracy=tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0],tf.int32),targets))

#Read Data
#Input
train_image_list,train_gt_list,keys=utils.getlist("train.txt")
train_images,train_sequence_length=utils.readimages(train_image_list)
train_gt_all=utils.readgt(train_gt_list)

#Valid
#valid_image_list,valid_gt_list=getlist("valid.txt")
#valid_images,valid_sequence_length=readimages(valid_image_list)
#valid_gt_all=readgt(valid_gt_list)

def do_batch(input_image,target,seq_length):
    feed={inputs:input_image,targets:target,seq_len:seq_length}
    batch_cost,steps,_=session.run([cost,global_step,optimizer],feed)

    return batch_cost,steps

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(graph=train,config=config) as session:
    ckpt=tf.train.get_checkpoint_state("models")
    if ckpt and ckpt.model_checkpoint_path:
        saver=tf.train.Saver()
        saver.restore(session,ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found, training from scratch")
        #Initialize the variables
        session.run(tf.global_variables_initializer())
        saver=tf.train.Saver(tf.all_variables(),max_to_keep=100)
    for curr_epoch in range(utils.num_epochs):
        print("Current Epoch: ",curr_epoch)
        train_cost=0
        for batch in range(utils.batches):
            start=time.time()
            current_batch=keys[batch*utils.batch_size:(batch+1)*utils.batch_size]
            curr_images=list()
            curr_gt=list()
            curr_seq_len=list()

            for a in current_batch:
                curr_images.append(train_images[a])
                curr_gt.append(train_gt_all[a])
                curr_seq_len.append(train_sequence_length[a])
            curr_images=np.array(curr_images)
            curr_gt=np.array(curr_gt)
            curr_seq_len=np.array(curr_seq_len)
            b_cost, steps=do_batch(curr_images,curr_gt,curr_seq_len)
            train_cost+=b_cost*utils.batch_size
            seconds=time.time()-start
            print("Steps: ",steps,", batch_second: ",seconds,
                  ", batch_cost: ",batch_cost)
