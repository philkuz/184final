from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt
from glob import glob
from params import *
from utils import *


try:
    from StringIO import StringIO
except ImportError:
    # from io import StringIO
    from io import BytesIO as StringIO

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer
def normal_initializer():
    return tf.truncated_normal_initializer(stddev=0.1)
initializer_map = {
    'identity': identity_initializer,
    'normal' : normal_initializer
}

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

def build(input, initializer=identity_initializer):
    net=slim.conv2d(input,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv1')
    net=slim.conv2d(net,24,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv2')
    net=slim.conv2d(net,24,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv3')
    net=slim.conv2d(net,24,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv4')
    net=slim.conv2d(net,24,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv5')
    net=slim.conv2d(net,24,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv6')
    net=slim.conv2d(net,24,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv7')
#    net=slim.conv2d(net,24,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv8')
    net=slim.conv2d(net,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=initializer(),scope='g_conv9')
    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_conv_last')
    return net

def prepare_data(train_file, test_file):
    input_files=[]
    output_files=[]
    val_input_files=[]
    val_output_files=[]
    with open(train_file) as f:
      for line in f:
          line = line.strip().split(',')
          input_files.append(line[0])
          output_files.append(line[1])
    with open(test_file) as f:
      for line in f:
          line = line.strip().split(',')
          val_input_files.append(line[0])
          val_output_files.append(line[1])
    return input_files, output_files, val_input_files, val_output_files

args = setup_argparse()
if args.params:
    params = read_params(args.params)
else:
    params = {}
default_params(params)
if params['gpu_number'] is None:
  os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
  os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
  os.system('rm tmp')
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu_number'])
is_training=params['is_training']
sess=tf.Session()
train_file = params['train_file']
test_file = params['test_file']
input_files, output_files, val_input_files, val_output_files = prepare_data(train_file, test_file)
input_img=tf.placeholder(tf.float32,shape=[None,None,None,3])
output=tf.placeholder(tf.float32,shape=[None,None,None,3])
if params['initializer'] not in initializer_map:
    raise ValueError('Initializer "{}" not available'.format(params['initializer']))
weights_initializer = initializer_map[params['initializer']]

if params['resize_size'] is not None:
    resize_img = tf.image.resize_images(input_img, params['resize_size'])
    network=build(resize_img, weights_initializer)
else:
    network=build(input_img, weights_initializer)
def get_loss(loss_type):
    if loss_type == 'mse':
      loss=lambda net, gt: tf.reduce_mean(tf.square(net-gt))
    elif loss_type == 'l1':
      loss = lambda net, gt: tf.reduce_mean(tf.abs(net-gt))
    else:
      raise ValueError("'{}' not implemented as a loss".format(loss_type))
    return loss

objective_loss = loss = get_loss('mse')(network,output)
if params['use_style']:
  import neural_style as ns
  ns.initialize_with_defaults()

  net_output = ns.build_vgg(ns.vgg_demean(network), True)
  net_style = ns.build_vgg(ns.vgg_demean(output), True)
  style_loss = ns.style_loss_arb(sess, net_output, net_style)
  loss = loss + params['style_weight'] * style_loss
global_step = tf.get_variable(
    'global_step', [], trainable = False,
    initializer = tf.constant_initializer(0), dtype = tf.int64)
opt=tf.train.AdamOptimizer(learning_rate=params['lr']).minimize(loss,var_list=[var for var in tf.trainable_variables()], global_step =global_step)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
# make a new directory everytime we run this
if params['continue_checkpoint'] is not None:
    logs_path = params['continue_checkpoint']
    ckpt = tf.train.get_checkpoint_state(logs_path)
    if ckpt:
        print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        raise ValueError('checkpoint in {} is invalid'.format(params['continue_checkpoint']))
    starting_epoch = int(sorted(glob(pj(logs_path, '[0-9]*')))[-1].split('/')[-1]) + 1


else:
    logs_path= params['logs_path']
    existing_logs = sorted(glob(pj(logs_path, '[0-9]*')))
    if len(existing_logs) == 0:
        logs_path = pj(logs_path, '{num:03d}'.format(num=0))
    else:
        highest_num= int(existing_logs[-1].split('/')[-1])
        logs_path = pj(logs_path, '{num:03d}'.format(num=highest_num + 1))
    os.makedirs(logs_path)
    if params['load_checkpoint']:
        if params['checkpoint'] is not None:
            ckpt=tf.train.get_checkpoint_state(params['checkpoint'])
            if ckpt:
                print('loaded '+ckpt.model_checkpoint_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                raise ValueError('checkpoint parameter "{}" is invalid'.format(params['checkpoint']))
        else:
            raise ValueError('checkpoint parameter "{}" is invalid'.format(params['checkpoint']))
    starting_epoch = 1
print('starting epoch', starting_epoch)
print(logs_path)
write_params(params, logs_path)
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

def pad(image, shape=(480, 480)):
    cur_shape = image.shape[-3:-1]
    zeros_dims = (np.array(shape) - np.array(cur_shape)).astype(np.float)
    zeros_dims /= 2.
    padding_array= [(0,0)]*len(image.shape[:-3]) + \
        [(int(np.floor(zeros_dims[0])), int(np.ceil(zeros_dims[0])))] + \
        [(int(np.floor(zeros_dims[1])), int(np.ceil(zeros_dims[1])))] + [(0,0)]
    return np.pad(image, padding_array, 'constant')

    return image

def merge_images(list_of_image_lists, out_shape=(480, 480)):
    ''' merge images '''

    numcols = len(list_of_image_lists)
    numrows = len(list_of_image_lists[0])
    lengths = [ len(l) for l in list_of_image_lists]
    all([l == numrows for l in lengths])
    assert all([l == numrows for l in lengths]), 'Not all lists are the same length: ' + (", ".join(["'{}'"]* numcols)).format(lengths)
    # TODO add a height and width parameter in the params
    height, width =  600, 800
    out_img = np.zeros((height*numrows, width*numcols, 3))
    for col in range(numcols):
        for row in range(numrows):
            image = list_of_image_lists[col][row]
            #TODO fix padding
            # pad image to 480x480
            image = pad(image, shape=(height, width))
            # print(image.shape)
            y0 = col * width
            y1 = y0 + width
            x0 = row * height
            x1 = x0 + height
            # print(x0, x1, y0, y1, out_img.shape)
            out_img[x0:x1, y0:y1] = image
    return out_img

def log_img(writer, tag, img, step):
    """Logs a list of images."""

    im_summaries = []
    # Write the image to a string
    s = StringIO()
    plt.imsave(s, img[:, :, ::-1], format='png')

    # Create an Image object
    img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                               height=img.shape[0],
                               width=img.shape[1])
    # Create a Summary value
    im_summaries.append(tf.Summary.Value(tag=tag,
                                         image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=im_summaries)
    writer.add_summary(summary, step)

def log_images(writer, tag, images, step):
    """Logs a list of images."""

    im_summaries = []
    for nr, img in enumerate(images):
        # Write the image to a string
        s = StringIO()
        plt.imsave(s, img, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                             image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=im_summaries)
    writer.add_summary(summary, step)

def apply_texture_queue(input_image, output_image, texture_queue_type):
    queue_types = ['center_circle']
    if texture_queue_type not in queue_types:
        raise ValueError('Texture queue type "{}" not recognized. Options are: {}'.format(texture_queue_type, queue_types))
    if texture_queue_type == 'center_circle':
        shape = input_image.shape[1:3]
        center = get_center(shape)
        radius = 20
        mask = circle_mask(shape, center, radius)
    queued_img = make_texture_queue(input_image, output_image, expand_dims_list(mask, [0,-1]))
    return queued_img


if params['is_training']:
    if params['use_style']:
      tf.summary.scalar('style_loss', style_loss, collections=['train'])
      tf.summary.scalar('objective_loss', objective_loss, collections=['train'])
    tf.summary.scalar('loss', loss, collections=['train'])
    train_summary_op = tf.summary.merge_all('train')
    all_losses=np.zeros(len(input_files), dtype=float)
    print(len(input_files))


    for epoch in range(starting_epoch,params['num_epochs'] +1):
        if epoch==starting_epoch:# or epoch==151:
            input_images=[None]*len(input_files)
            output_images=[None]*len(input_files)

        # if os.path.isdir("%s/%04d"%(task,epoch)):
        #     continue
        cnt=0
        random_files = np.random.permutation(len(input_files))
        if args.debug:
          random_files[:2]
        for id in random_files:
            step= sess.run([global_step])[0]
            st=time.time()
            if input_images[id] is None:
                input_images[id]=np.expand_dims(np.float32(cv2.imread(input_files[id] ,-1)),axis=0)/255.0
                output_images[id]=np.expand_dims(np.float32(cv2.imread(output_files[id],-1)),axis=0)/255.0
            # TODO check to make sure this works and is necessary
            if input_images[id].shape[1]*input_images[id].shape[2]>2200000:#due to GPU memory limitation
                continue
            if params['texture_queue'] is not None:
                input_images[id] = apply_texture_queue(input_images[id], output_images[id], params['texture_queue'])
            if epoch == 1:
                print(input_files[id], output_files[id])
            _, current, summary_ret =sess.run([opt,loss, train_summary_op],feed_dict={input_img:input_images[id],output:output_images[id]})
            writer.add_summary(summary_ret, step)
            all_losses[id]=current*255.0*255.0
            cnt+=1
            print("%d %d %.2f %.2f %.2f"%(epoch,cnt,current*255.0*255.0,np.mean(all_losses[np.where(all_losses)]),time.time()-st))
        epoch_dir = pj(logs_path, "%04d" %epoch)
        if not os.path.exists(epoch_dir):
            # os.rmdir(epoch_dir)

            os.makedirs(epoch_dir)
        with open(pj(epoch_dir, 'score.txt'), 'w') as target:
            target.write("%f"%np.mean(all_losses[np.where(all_losses)]))

        saver.save(sess,pj(epoch_dir, 'model.ckpt'))
        saver.save(sess,pj(logs_path, 'model.ckpt'), global_step = global_step)
        val_input_images = []
        val_output_gts = []
        val_output_pred = []
        for ind in range(10):
            input_image=np.expand_dims(np.float32(cv2.imread(val_input_files[ind],-1)),axis=0)
            output_image_gt=np.expand_dims(np.float32(cv2.imread(val_output_files[ind],-1)),axis=0)
            if params['texture_queue'] is not None:
                input_image = apply_texture_queue(input_image, output_image_gt, params['texture_queue'])
            val_input_images.append(input_image)
            val_output_gts.append(output_image_gt)
            st=time.time()
            output_image=sess.run(network,feed_dict={input_img:input_image/255.0})
            print("ffwd time %.3f"%(time.time()-st))
            output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
            val_output_pred.append(output_image)
            if params['val_triples']:
                output_image = merge_images([[input_image], [output_image], [output_image_gt]])
                cv2.imwrite(pj(epoch_dir,"%06d.jpg"%(ind)),np.uint8(output_image))
            else:
                cv2.imwrite(pj(epoch_dir,"%06d.jpg"%(ind)),np.uint8(output_image[0,:,:,:]))
        merged_image = merge_images([val_input_images,  val_output_pred, val_output_gts])
        cv2.imwrite(pj(epoch_dir, 'all.jpg'), np.uint8(merged_image))
        # log_img(writer, 'all', pj(epoch_dir, 'all.jpg'),  step)
        log_img(writer, 'all', merged_image/255.,  step)

out_dir = pj(logs_path, 'val_output')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for val_file, output_file in zip(val_input_files, val_output_files):
    if not os.path.isfile(val_file):
        continue
    contain_dir, _= val_file.split('/')[-2:]
    _, filename = output_file.split('/')[-2:]
    #filename, ext = filename.split('.')
    #filename = '/'.join((contain_dir, filename)
    # print(contain_dir, filename)
    if not os.path.exists(os.path.join(out_dir, contain_dir)):
        os.makedirs(os.path.join(out_dir, contain_dir))

    read_image = cv2.imread(val_file,-1)
    output_image_gt = cv2.imread(output_file,-1)

    # scale = 480/max(read_image.shape) # * max(read_image.shape)
    # read_image = cv2.resize(read_image, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

    input_image=np.expand_dims(np.float32(read_image),axis=0)/255.0
    output_image_gt=np.expand_dims(np.float32(output_image_gt),axis=0)/255.0
    st=time.time()
    if params['texture_queue'] is not None:
        input_image = apply_texture_queue(input_image, output_image_gt, params['texture_queue'])
    output_image=sess.run(network,feed_dict={input_img:input_image})
    print("%.3f"%(time.time()-st))
    output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
    if params['val_triples']:
        output_image = merge_images([[input_image*255], [output_image], [output_image_gt*255]])
        cv2.imwrite(os.path.join(out_dir, filename) ,np.uint8(output_image))
    else:
        cv2.imwrite(os.path.join(out_dir, filename) ,np.uint8(output_image[0,:,:,:]))

