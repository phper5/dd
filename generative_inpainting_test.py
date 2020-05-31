import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from generative_inpainting.inpaint_model import InpaintCAModel


def test(image_file,mask_file,output_path):

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=image_file, type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default=mask_file, type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default=output_path, type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='generative_inpainting/checkpoints', type=str,
                        help='The directory of tensorflow checkpoint.')


    FLAGS = ng.Config('generative_inpainting/inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    # b_channel, g_channel, r_channel = cv2.split(mask)
    # mask = cv2.merge((b_channel, g_channel, r_channel, r_channel))
    # cv2.imwrite(args.mask, mask)
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
    # if image.shape[2]==3:
    #     b_channel, g_channel, r_channel = cv2.split(image)
    #     alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    #     image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    #     cv2.imwrite(args.image,image)
    if image.shape[2] == 4:
        b_channel, g_channel, r_channel = cv2.split(mask)
        mask = cv2.merge((b_channel, g_channel, r_channel, r_channel))
        cv2.imwrite(args.mask, mask)
    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
    tf.reset_default_graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(args.output, result[0][:, :, ::-1])
if __name__ == "__main__":
    test('/data/code/python/diandi/2020-05-09-19/3035cb1e-d617-30fb-9762-92281c4d08af-max-img.png','/data/code/python/diandi/2020-05-09-19/3035cb1e-d617-30fb-9762-92281c4d08af-max-mask.png','/home/white/Desktop/case1_output.png');