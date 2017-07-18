import os
import cv2
import codecs
import numpy as np

DIGITS = "~!%'()+,-.\/0123456789:ABCDEFGIJKLMNOPRSTUVWYabcdefghiklmnoprstuvwxz-V،د‘“ ؤب,گ0ذصط3وLِbT2dh9ٰٴxAڈlژ؛؟أGاpث4/س7ًtCهKیُS\"۔WOcgk…ٓosw(ﷺجڑ.آئکتخز6غEشہقنضDNR8ظ:fnrvzپچB’”لء%)ْFحر5عںھف!JمIM#ّےUYَae'Pimة1uٹ+"

#Hyper parameters
window_height=48
num_classes=len(DIGITS)+1 #Number of classes plus blank class for ctc
learning_rate_decay_factor=0.9
momentum=0.9
initial_learning_rate=1e-4
decay_steps=5000

batch_size=100
test_batch_size=100
batches=100
train_size=batch_size*batches

num_epochs=100
num_hidden=256
num_layer=1

def getlist(path):
    images=open(path,'r').readlines()
    imagelist=list()
    gtlist=list()
    keys=list()
    for a in images:
        a=a.strip('\n')
        b=os.path.splitext(os.path.splitext(a)[0])[0]
        keys.append(b)
        b=b+'.gt.txt'
        imagelist.append(a)
        gtlist.append(b)
    return imagelist,gtlist,keys

def readimages(imagelist):
    maxlength=0
    images={}
    seqlength={}
    for a in imagelist:
        image=cv2.imread(a,0)
        a=os.path.splitext(a)[0]
        h,w=image.shape
        w=int(w*(window_height/(h*1.)))
        image=cv2.resize(image,(w,window_height))
        h,w=image.shape
        if(w>maxlength):
            maxlength=w
        images[a]=image
    for key in images:
        image=images[key]
        h,w=image.shape
        seqlength[key]=w
        border=maxlength-w
        image=cv2.copyMakeBorder(image, 0, 0, 0, border, borderType= cv2.BORDER_CONSTANT, value=[255] )
        images[key]=image
    return images,seqlength

def readgt(gtlist):
    indices=[]
    values=[]
    groundtruth={}
    maxsent=0
    for ind,current_gt in enumerate(gtlist):
        gt=codecs.open(current_gt,'r','utf-8').readline()
        print(gt)
        if(len(gt)>maxsent):
            maxsent=len(gt)
        current_gt=os.path.splitext(os.path.splitext(current_gt)[0])[0]
        groundtruth[current_gt]=gt
        for idx,curr_char in enumerate(list(gt)):
            values.append(DIGITS.find(curr_char))
            indices.append([ind,idx])
    indices=np.array(indices)
    values=np.array(values)
    dense_shape=[len(gtlist),maxsent]
    return (indices,values,dense_shape)

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = common.DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []

    for idx, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(idx)

    decoded_indexes.append(current_seq)
    result = []

    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))

    return result
