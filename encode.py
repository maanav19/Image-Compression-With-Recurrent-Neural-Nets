from __future__ import print_function
import io
import numpy as np
from imageio import imread, imsave
import tensorflow as tf
from model import encoder, decoder

#Flags

tf.compat.v1.flags.DEFINE_string(name= 'input', default= None, help= 'image path input')
tf.compat.v1.flags.DEFINE_integer(name= 'iters', default= 10, help= 'number of iterations')
tf.compat.v1.flags.DEFINE_string(name= 'output', default= 'compressed', help= 'output path')
tf.compat.v1.flags.DEFINE_string(name= 'model', default= 'save/model', help= 'saved model')


try:
    image = imread(tf.compat.v1.flags.FLAGS.input).astype(tf.float32)

except:
    print('please check the image path')
    exit()

height, width, channel = image.shape
new_height = height + 16 - height % 16
new_width = width + 16 - height % 16
pad_height = new_height - height
pad_width = new_width - width
img_padded = np.pad(image, ((0, pad_height), (0, pad_width), (0,0)), 'constant')

inputs = np.expand_dims(img_padded, axis= 0)
batch_size = 1

pinputs = tf.compat.v1.placeholder(tf.float32, [batch_size, new_height, new_width, 3])
inputs_ = (pinputs / 255) - 0.5

enc = encoder(batch_size= batch_size, height= new_height, width= new_width)
dec = decoder(batch_size= batch_size, height= new_height, width= new_width)
codes = []

for i in range(tf.compat.v1.flags.FLAGS.iters):
    code = enc.encode(inputs_)
    codes.append(code)
    outputs = dec.decode(code)
    inputs_ = inputs_ - outputs

save = tf.compat.v1.train.Saver()

eval_codes = []
with tf.compat.v1.Session():
    save.restore(sess= tf.compat.v1.Session, tf.compat.v1.flags.FLAGS.model)
    for j in range(tf.compat.v1.flags.FLAGS.iters):
        c = codes[j].eval(feed_dict={pinputs: inputs})
        eval_codes.append(c)

int_codes = (np.stack(eval_codes).astype(tf.int8) + 1) // 2
export = np.packbits(int_codes.reshape(-1))

np.savez_compressed(tf.compat.v1.flags.FLAGS.output, s= int_codes.shape, o= (height, width), c= export)

