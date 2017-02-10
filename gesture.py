import os,sys
import find_mxnet
import mxnet as mx
import logging
import time
import cv2
import random
import glob
import numpy as np
import string
import cPickle as p
from PIL import Image,ImageDraw,ImageFont

aaaa = 0

BATCH_SIZE = 1
LEN_SEQ = 10

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n,x in zip(self.label_names, self.label)]

def readData(Filename, data_shape):
    data_1 = []
    data_2 = []

    pic = []
    pic_x = []
    pic_y = []
    for filename in glob.glob(Filename+'/image*.jpg'):
        pic.append(filename)
    pic.sort()
    a = 0
    for i in range(len(pic)-1):
        prev = cv2.imread(pic[i], cv2.IMREAD_GRAYSCALE)
	prev = cv2.resize(prev, (data_shape[2], data_shape[1]/10))
	#prev = np.multiply(prev, 1/255.0)
        #print prev[156][0:100]

        cur = cv2.imread(pic[i+1], cv2.IMREAD_GRAYSCALE)
	cur = cv2.resize(cur, (data_shape[2], data_shape[1]/10))
	#cur = np.multiply(cur, 1/255.0)
	#print cur.shape
        #print cur[156][0:100]

        flow  = cv2.calcOpticalFlowFarneback(prev, cur, 0.702, 5, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
	#flow = np.array(flow)
	#flow = np.multiply(flow, 255)
	array_bound = np.ones((data_shape[2], data_shape[1]/10, 2), dtype=int)*20
	flow_img = 255*((flow)+array_bound)/(2*20)
        flow_img = np.uint8(flow_img)
	#aaaa += 1
	#flow = np.multiply(flow_img, 1/255.0)
	
	flow_1 = flow_img.transpose((2,0,1))
	#cv2.imwrite('./flow/flow-'+str(a)+'.jpg',flow_1[0,...])
	flow_1 = flow_1.tolist()
	pic_x.append(flow_1[0])
	pic_y.append(flow_1[1])
	a += 1
    
    for j in range(len(pic_x)-LEN_SEQ):
        data_1_1 = []
        for i in range(LEN_SEQ):
	    idx = j+i
            data_1_1.append([pic_x[idx], pic_y[idx]]) 
        data_2.append(0)
        data_1.append(data_1_1)
    return (data_1, data_2)

def readImg(Filename, data_shape):
    mat = []
    img_1 = Filename[0][0]
    img_2 = Filename[0][1]

    for i in range(LEN_SEQ-1):
        tmp_1 = Filename[i+1][0]
	img_1.extend(tmp_1)

        tmp_2 = Filename[i+1][1]
	img_2.extend(tmp_2)

    mat.append(img_1)
    mat.append(img_2)

    return mat

class GestureIter(mx.io.DataIter):
    def __init__(self, fname, batch_size, seq_len, data_shape, init_states):
        self.batch_size = batch_size
        self.fname = fname
	self.seq_len = seq_len
        self.data_shape = data_shape
        (self.data_1, self.data_3) = readData(self.fname, self.data_shape)
        self.num = len(self.data_1)/batch_size
        print len(self.data_1)
        #print self.data_1

	self.init_states = init_states
	self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size,) + data_shape)] + init_states
        self.provide_label = [('label', (batch_size, seq_len))]

    def __iter__(self):
        init_states_names = [x[0] for x in self.init_states]
        for k in range(self.num):
            data = []
            label = []
            for i in range(self.batch_size):
                idx = k * self.batch_size + i
                img = readImg(self.data_1[idx], self.data_shape)
		#print len(img), len(img[0])
                data.append(img)
		label_tmp = []
		for i in range(self.seq_len):
		    label_tmp.append(self.data_3[idx])
                label.append(label_tmp)
                #label.append(self.data_3[idx])

            data_all = [mx.nd.array(data)]+self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data']+init_states_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':

    batch_size = BATCH_SIZE
    data_shape = (2,2560,256)

    num_hidden = 4096
    num_lstm_layer = 2 

    num_label = 5
    seq_len = LEN_SEQ

    devs = [mx.context.gpu(0)]

    test_file = './1'

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_val = GestureIter(test_file, batch_size, seq_len, data_shape, init_states)
    print data_val.provide_data, data_val.provide_label

    model = mx.model.FeedForward.load("./model/cnn_lstm", epoch=500, ctx=devs)

    internels = model.symbol.get_internals()

    fea_symbol = internels['fc_output']

    feature_exactor = mx.model.FeedForward(ctx=devs, symbol=fea_symbol, num_batch_size=1,
                                           arg_params=model.arg_params, aux_params=model.aux_params,
					   allow_extra_params=True)

    cnn_test_result = feature_exactor.predict(data_val)
    predict_result = []
    print np.array(cnn_test_result).shape
    print cnn_test_result[0]
    for i in range(len(cnn_test_result)):
        predict_result.append(np.argmax(cnn_test_result[i]))

    print predict_result

    pic = []
    for filename in glob.glob('./1/image*.jpg'):
        pic.append(filename)
    pic.sort()

    for i in range(len(pic)-11):
        idx = i+11
	dic = {0:'no', 1:'stable', 2:'wave', 3:'Positive rotation', 4:'Reverse roration'}
	ttfont = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf",50)
	im = Image.open(pic[idx])
	draw = ImageDraw.Draw(im)
	draw.text((10,10),dic[predict_result[i]], fill=(255,0,0),font=ttfont)
	im.save(pic[idx])
