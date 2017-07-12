import utils
import editdistance
import time
from model import get_train_model
from utils import decode_sparse_tensor



def report_accuracy(decoded_list,test_targets,test_names):
    original_list=decode_sparse_tensor(test_targets)
    detected_list=decode_sparse_tensor(decoded_list)
    names_list=test_names.tolist()

    total_edits=0
    total_length=0
    test_batches=len(detected_list/test_batch_size)
    for a in range(test_batches):
        curr_original_list=original_list[(a*test_batch_size):((a+1)*test_batch_size)]
        curr_detected_list=detected_list[(a*test_batch_size):((a+1)*test_batch_size)]
        if len(original_list)!= len(detected_list):
            print("The length of output list: ",len(original_list)," did not match the ground truth: ",len(original_list)," skipping this batch")
            return
        print("Ground Truth Length: ",len(original_list)," Output Length: ",len(detected_list))
        for idx,current_gt in enumerate(curr_original_list):
            current_output=curr_detected_list[idx]
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


#Read Data
#Input
train_image_list,train_gt_list=getlist("train.txt")
train_images,train_sequence_length=readimages(image_list)
train_gt_all=readgt(gt_list)

#Valid
valid_image_list,valid_gt_list=getlist("valid.txt")
valid_images,valid_sequence_length=readimages(valid_image_list)
valid_gt_all=readgt(valid_gt_list)

gpuoptions=tf.GPUOptions(per_process_gpu_memory=0.4)
