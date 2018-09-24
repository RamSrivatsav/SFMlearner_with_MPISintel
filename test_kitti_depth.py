from __future__ import division
import tensorflow as tf
import numpy as np
import os
# import scipy.misc
from PIL import Image
from SfMLearner import SfMLearner
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
from utils import normalize_depth_for_display

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 512, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("save_output_depthimg", None, 'saving link')
flags.DEFINE_string("save_output_depthdata", None, 'saving link')
flags.DEFINE_string("test_file_list", None, "test file list")
FLAGS = flags.FLAGS
test_files_txt=os.listdir(FLAGS.test_file_list)
def main():
    basename = os.path.basename(FLAGS.ckpt_file)
#        output_file = FLAGS.output_dir + '/' + basename
    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height, img_width=FLAGS.img_width, batch_size=FLAGS.batch_size, mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    for files in test_files_txt:
        newfolder = os.path.splitext(files)[0]
        savepathimage= FLAGS.save_output_depthimg + '/' + newfolder
        #print(savepathimage)
        if not os.path.exists(savepathimage):
            os.makedirs(savepathimage)
        savepathdata= FLAGS.save_output_depthdata + '/' + newfolder
        if not os.path.exists(savepathdata):
            os.makedirs(savepathdata)
        print(files)
        
        
        with open(os.path.join(FLAGS.test_file_list,files), 'r') as f:
            test_files = f.readlines()
            test_files = [t[:-1] for t in test_files]
#        if not os.path.exists(FLAGS.output_dir):
#            os.makedirs(FLAGS.output_dir)
        
        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.ckpt_file)
            pred_all = []
            for t in range(0, len(test_files), FLAGS.batch_size):
                if t % 100 == 0:
                    print('processing %s: %d/%d' % (basename, t, len(test_files)))
                inputs = np.zeros(
                    (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3), 
                    dtype=np.uint8)
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    print(len(test_files))
                    if idx >= len(test_files):
                        break
#                    fh = open(test_files[idx], 'r')
#                    print(fh)
#                    raw_im = Image.open(fh)
#                    scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), Image.ANTIALIAS)
#                    inputs[b] = np.array(scaled_im)
                    im = scipy.misc.imread(test_files[idx])
                    inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
                pred = sfm.inference(inputs, sess, mode='depth')
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    dept= pred['depth'][b,:,:,0]
                    imagesave = normalize_depth_for_display(dept)
                    plt.imsave(os.path.join(savepathimage , newfolder + '_'+ str(idx) + '.png'), imagesave)
                    pred_all.append(pred['depth'][b,:,:,0])
                    sio.savemat(os.path.join(savepathdata,newfolder + '_'+ str(idx)), {'D': dept.squeeze()})

if __name__ == '__main__':
    main()