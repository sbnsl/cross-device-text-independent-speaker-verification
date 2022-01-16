import  tensorflow as tf
import  numpy as np
import  sys
import glob
import  os
import  random
import math
import collections
import time
import shutil
# import matplotlib.pyplot as plt
########################################################################slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
slim = tf.contrib.slim

from towers import *

# Parameter setting ****************************************************************************************************
# directory of input samples
input_dir_train = '/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'
input_dir_test  = '/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'


# my_file = Path(file_P)
# if my_file.is_file():
#     os.remove(fiel_P)

# number of channels in input samples
n_input_ch = 3
lr=0.001

# mode of run
_Mode=2
mode ="train"
isLoadModel = True



if _Mode==0:
    mode='train'
    isLoadModel = False

elif _Mode==1:
    mode = 'train'
    isLoadModel = True
elif _Mode==2:
    mode='test'
    isLoadModel = True






#
if mode == "train":
    isTraining = True
else:
    isTraining = False



batch_size = 32
num_classes=411
reg_w=0.0009



# log_dir
log_dir ='./utterance3s/logs_gp_2mp_5c_regw00009_test/'
save_dir = './utterance3s/model_gp_2mp_5c_regw00009_test/'


# if os.path.isdir(log_dir_test):
#     shutil.rmtree(log_dir_test)


# max epoch
max_epoch = 200

# adam optimizer parameters
beta1 = 0.5 # momentum term of adam
initial_lr = 0.001 # initial learning rate
sync_replicas=0
replicas_to_aggregate=1
num_epochs_per_decay=2
learning_rate_decay_type='exponential'
learning_rate_decay_factor=0.95
end_learning_rate=0.00001



# display parameters

display_step = 20

# saving frequency

save_freq = 1000
summary_freq = 20
samplesize=[300,40,3]



# Collections definition ***********************************************************************************************
Examples = collections.namedtuple("Examples", " images, label, count, steps_per_epoch")
# Model = collections.namedtuple("Model", "pos_loss, neg_loss, distance")



def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        # return image * 2 - 1
        return image

def deprocess(image):
    with tf.name_scope("deprocess"):
        # return (image + 1) / 2
        return image


# new data loader
def read_labeled_image_list(image_list_file, img_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    clss = []
    for line in f:
        # print line
        filename, idx = line[:-1].split(' ')
        filenames.append(img_dir+'/'+filename)
        clss.append(int(idx))

    return filenames, clss

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    # file_contents = np.load(input_queue[0])

    # file_contents = tf.read_file(input_queue[0])
    # example = tf.image.decode_jpeg(file_contents, channels=3)


    file_contents = tf.read_file(input_queue[0])
    example = tf.decode_raw(file_contents, out_type=tf.float64)
    example=tf.reshape(example,samplesize)

    return example, label

def load_examples(datadir, lblfileaddress):
    filename = lblfileaddress
    imgdir = datadir
    # Reads pfathes of images together with their labels
    image_list,  label_class_list = read_labeled_image_list(filename, imgdir)
    # print image_list
    # print label_class_list
    print len(image_list)
    print len(label_class_list)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_class_list, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                shuffle=isTraining)

    raw_image, label = read_images_from_disk(input_queue)
    raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)
    assertion = tf.assert_equal(tf.shape(raw_image)[2], n_input_ch, message="image does not have required channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_image)

    raw_input.set_shape([None, None, n_input_ch]) # was 3

    # images = preprocess(raw_input)
    images = raw_input

    seed = random.randint(0, 2**31 - 1)


    # #scale and crop input image to match 256x256 size
    def transform(image):
        r = image
        # #r.set_shape([192,192,1])
        # # if a.flip:
        # #     r = tf.image.random_flip_left_right(r, seed=seed)
        #
        # # area produces a nice downscaling, but does nearest neighbor for upscaling
        # # assume we're going to be doing downscaling here
        # r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
        #
        # offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        # if scale_size > CROP_SIZE:
        #     r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        #
        # elif scale_size < CROP_SIZE:
        #     raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("images"):
        print images.get_shape()

        input_images = transform(images)

        # input_images = images



    # Optional Image and Label Batching
    image_batch, label_batch = tf.train.batch([input_images, label],
                                              batch_size=batch_size)

    steps_per_epoch = int(math.ceil(len(label_class_list) / batch_size))



    return Examples(
        images=image_batch,
        label=label_batch,
        count=len(label_class_list),
        steps_per_epoch=steps_per_epoch,
    )



def load_utterance(datadir, lblutteranceaddress):
    filename = lblutteranceaddress
    imgdir = datadir
    # Reads pfathes of images together with their labels
    image_list,  label_class_list = read_labeled_image_list(filename, imgdir)
    # print image_list
    # print label_class_list
    print len(image_list)
    print len(label_class_list)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_class_list, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                shuffle=isTraining)

    raw_image, label = read_images_from_disk(input_queue)
    raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)
    assertion = tf.assert_equal(tf.shape(raw_image)[2], n_input_ch, message="image does not have required channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_image)

    raw_input.set_shape([None, None, n_input_ch]) # was 3

    # images = preprocess(raw_input)
    images = raw_input

    seed = random.randint(0, 2**31 - 1)


    # #scale and crop input image to match 256x256 size
    def transform(image):
        r = image
        # #r.set_shape([192,192,1])
        # # if a.flip:
        # #     r = tf.image.random_flip_left_right(r, seed=seed)
        #
        # # area produces a nice downscaling, but does nearest neighbor for upscaling
        # # assume we're going to be doing downscaling here
        # r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
        #
        # offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        # if scale_size > CROP_SIZE:
        #     r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        #
        # elif scale_size < CROP_SIZE:
        #     raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("images"):
        print images.get_shape()

        input_images = transform(images)

        # input_images = images



    # Optional Image and Label Batching
    image_batch, label_batch = tf.train.batch([input_images, label],
                                              batch_size=batch_size)

    steps_per_epoch = int(math.ceil(len(label_class_list) / batch_size))



    return Examples(
        images=image_batch,
        label=label_batch,
        count=len(label_class_list),
        steps_per_epoch=steps_per_epoch,
    )







def create_model(examples,num_classes,isTraining):
    with tf.variable_scope("utils"):
        pred  = create_tower_2mp_5c(examples.images,num_classes,isTraining)
        # print pred.get_shape
        # print camconvA.get_shape()
    return pred

def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / batch_size *
                      num_epochs_per_decay)
    if sync_replicas:
        decay_steps /= replicas_to_aggregate

    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(initial_lr,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        return tf.constant(initial_lr, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(initial_lr,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def Wdecay():
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find('weights') > 0:
            costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)




def main():

    if isTraining:
        lblfileaddress = "/home/Speaker/Speech/Classification_Utternace/classification_sample_train"
        lblutteranceaddress = '/home/Speaker/Speech/Classification_Utternace/classification_utterance_train_map'

        datadir = input_dir_train
    else:
        lblfileaddress = "/home/Speaker/Speech/Classification_Utternace/classification_sample_test"
        lblutteranceaddress = '/home/Speaker/Speech/Classification_Utternace/classification_utterance_test_map'

        datadir = input_dir_test
    examples = load_examples(datadir, lblfileaddress)


    # input_dir = input_dir_train
    #
    # examples = load_examples(input_dir)
    print(">>>>> examples count = %d" % examples.count)

    num_samples=examples.count


    pred = create_model(examples,num_classes,isTraining)


    print pred.get_shape()
    # print camconv.get_shape()


    pred_argmax = tf.argmax(pred, axis=1)
    lbl = tf.cast(examples.label, tf.int64)

    acc = tf.equal(lbl, pred_argmax)
    acc = tf.cast(acc, tf.float32)
    acc = tf.reduce_mean(acc)




    with tf.name_scope("loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=examples.label, logits=pred)#tf.abs(pred - examples.offset)
        loss_xent = tf.reduce_mean(loss)
        loss_reg=Wdecay()*reg_w
        # loss_reg = 0.9 * tf.add_n(tf.losses.get_regularization_losses())
        loss_total = loss_xent+loss_reg

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        optim = tf.train.AdamOptimizer(lr, beta1)
        train_op = optim.minimize(loss_total)
        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        # lr = _configure_learning_rate(num_samples, global_step)



        train = tf.group(train_op, incr_global_step)

        # Ensures that we execute the update_ops before performing the train_step
        # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


    # with tf.name_scope("train"):
    #     optim = tf.train.AdamOptimizer(lr, beta1)
    #     # grads_and_vars = optim.compute_gradients(loss)
    #     # train = optim.apply_gradients(grads_and_vars)
    #     train_op = optim.minimize(loss)
    # global_step = tf.contrib.framework.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step, global_step + 1)
    # train = tf.group(train_op, incr_global_step)


    # # rotate images
    # rotated = tf.contrib.image.rotate

    input_images = deprocess(examples.images)
    # print input_images.get_shape
    # cropA_d = deprocess(examples.cropA)
    # cropB_d = deprocess(examples.cropB)
    # input_images_deformed = deprocess(examples.images_deformed)


    tf.summary.image("AAinput", input_images)
    # # tf.summary.image("AAinput_deformed", input_images_deformed)
    # # tf.summary.image("cropA", cropA_d)
    # # tf.summary.image("cropB", cropB_d)
    # tf.summary.image("cam", cam)
    # tf.summary.image("cammask", cam_msk )
    # tf.summary.image("camimpose", cam_msk*input_images )
    # # tf.summary.image("camB", camB)
    # # masked = cropB_d * camB
    # # tf.summary.image("masked", masked)

    tf.summary.scalar("loss_total", loss_total)
    tf.summary.scalar("loss_xent", loss_xent)
    tf.summary.scalar("loss_reg", loss_reg)
    # tf.summary.scalar("loss_cam", loss_cam)
    tf.summary.scalar("accuracy", acc)
    tf.summary.scalar(("learning rate"),lr)


    saver = tf.train.Saver(max_to_keep=10)
    # vars = [var for var in tf.trainable_variables() if var.name.startswith("verifier")]
    # saver1 = tf.train.Saver(vars)

    sv = tf.train.Supervisor(logdir=log_dir, save_summaries_secs=120, saver=None)
    with sv.managed_session() as sess:


        if isLoadModel:
            print ("loading from checkpoint...")
            checkpoint = tf.train.latest_checkpoint(save_dir)
            saver.restore(sess, checkpoint)
         #    print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
            # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     print i



            # raw_input("input")


        max_steps = 2**32
        if max_epoch is not None:
            max_steps = examples.steps_per_epoch * max_epoch
            print "max epochs: ", max_epoch
            print "max steps : ", max_steps
            start = time.time()


        n = 0
        acc_total = 0

        if isTraining:
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
                fetches = {
                    "train" : train,
                    "global_step" : sv.global_step,
                    "loss" : loss_total,
                    "labels" : examples.label,
                    "acc" : acc,
                    "ex" : examples.images,
                }

                if should(freq=summary_freq):
                    fetches["summary"] = sv.summary_op
                results = sess.run(fetches)

                acc_total = acc_total + results["acc"]
                n = n+1.

                if should(freq=summary_freq):
                    #print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                    a = np.amin( results["ex"], axis=(1, 2, 3))
                    b = np.amax( results["ex"], axis=(1, 2, 3))
                    print results["labels"]
                    # print a.shape
                    # print a
                    # print results["ex"].max(0)
                    #print results["loss"]
                if should(freq=display_step):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * batch_size / (time.time() - start)
                    remaining = (max_steps - step) * batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("loss", results["loss"])

                    acc_total = acc_total*1./n

                    print ("acc", acc_total)
                    acc_total = 0.
                    n = 0.
                    # print results["offset"]


                if should(freq=save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(save_dir, "model"), global_step=sv.global_step)
        else:

            # file1 = open(file_P, 'w')



            P=[]
            max_steps = examples.count//batch_size

            # max_steps = 100

            ac = 0.
            lbl_list = []
            pred_list = []
            for step in range(max_steps):
                acc_, pred0_,pred_, lbl_ = sess.run([acc,pred, pred_argmax, examples.label])

                # for item in pred0_:
                    # print>> file1, item


                if step%10==0:
                    print('step= ', step)
                ac = ac + acc_
                if step==0:
                    P=pred0_
                else:
                    P=np.concatenate((P,pred0_),axis=0)
                lbl_list.append(lbl_)
                pred_list.append(pred_)
            print "Total acc:", ac/max_steps*1.
            print len(lbl_list)
            # print lbl_list[10]
            lbl_array = np.asarray(lbl_list)
            pred_array = np.asarray(pred_list)
            lbl_array = np.reshape(lbl_array, [-1])
            pred_array = np.reshape(pred_array, [-1])
            print lbl_array.shape
            print pred_array.shape
            ac_utterance=[]
            ac_utterance3 = []
            ac_utterance5 = []
            ac_utterance10 = []
            ac_utterance100 = []
            for i in range(num_classes):
                idx = np.where(lbl_array==i)
                # print idx
                lbl_idx = lbl_array[idx]
                pred_idx = pred_array[idx]
                # print pred_idx
                P_=P[idx]
                p_utterance = np.sum(P_,axis=0)
                pred_ut=np.argmax(p_utterance)
                ac_utterance.append(1*(pred_ut==i))

                v=p_utterance.argsort()[-3:][::-1]
                ac_utterance3.append(1*(i in v))

                v = p_utterance.argsort()[-5:][::-1]
                ac_utterance5.append(1 * (i in v))

                v = p_utterance.argsort()[-10:][::-1]
                ac_utterance10.append(1 * (i in v))

                v = p_utterance.argsort()[-100:][::-1]
                ac_utterance100.append(1 * (i in v))

                correct = np.equal(pred_idx, lbl_idx )
                # print correct
                correct = correct.astype(float)

                print "acc ", i, " =", np.mean(correct), " - number of samples: ", lbl_idx.shape[0]

            ac_u=np.asarray(ac_utterance)
            ac_u = ac_u.astype(float)

            print('accuracy ut= ',(np.mean(ac_u)))


            ac_u3=np.asarray(ac_utterance3)
            ac_u3 = ac_u3.astype(float)

            print('accuracy ut3= ',np.mean(ac_u3))

            ac_u5=np.asarray(ac_utterance5)
            ac_u5 = ac_u5.astype(float)

            print('accuracy ut5= ',np.mean(ac_u5))

            ac_u10=np.asarray(ac_utterance10)
            ac_u10 = ac_u10.astype(float)

            print('accuracy ut10= ',np.mean(ac_u10))


            ac_u100=np.asarray(ac_utterance100)
            ac_u100 = ac_u100.astype(float)

            print('accuracy ut100= ',np.mean(ac_u100))


main()


