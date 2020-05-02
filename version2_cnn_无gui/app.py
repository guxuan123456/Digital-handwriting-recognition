import os
import glob
#import tensorflow as tf
import mnist_inference
import mnist_train
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def evaluate(pic, pic_name):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x:[pic]}
        y = mnist_inference.inference(x, None) #神经网络前向传播结果，这里的y相当于预测值
        result = tf.argmax(y, 1) #取最大值作为输出结果
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        '''
        -  输入层节点数，对于MNIST数据集，这个就等于图片的像素。
            INPUT_NODE = 784
        - 输出层节点数，对应数字0~9
            OUTPUT_NODE = 10
        - 隐藏层节点数，这里使用只有一个隐藏层的网络结构，节点有经验公式可以计算
            LAYER1_NODE = 500

        tf.train.ExponentialMovingAverage这个函数用于更新参数，就是采用滑动平均的方法更新参数。这个函数初始化需要提供一个衰减速率（decay），用于控制模型的更新速度。这个函数还会维护一个影子变量（也就是更新参数后的参数值），这个影子变量的初始值就是这个变量的初始值，影子变量值的更新方式如下：

        shadow_variable = decay * shadow_variable + (1-decay) * variable
        '''
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                number = sess.run(result, feed_dict=validate_feed)
                pic_name = pic_name.split('\\')[-1]
                print(pic_name,' is :',number[0])
            else:
                print('No checkpoint file found')
                return

def image_prepare(pic_name): 
    # 读取图像，第二个参数是读取方式
    img = cv2.imread(pic_name, 1)
    # 使用全局阈值，降噪
    
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # 把opencv图像转化为PIL图像
    im = Image.fromarray(cv2.cvtColor(th1,cv2.COLOR_BGR2RGB))
    # 灰度化
    im = im.convert('L')
    # 为图片重新指定尺寸
    im = im.resize((28,28), Image.ANTIALIAS)
    # plt.imshow(im)
    # plt.show()
    # 图像转换为list
    im_list = list(im.getdata())

    print(im_list)
    # 图像灰度反转
    result = [(255-x)*1.0/255.0 for x in im_list]
    return result


def main(argv=None):
    # 把要识别的图片放到下面的文件夹中
    img_path = 'picture/'
    imgs = glob.glob(os.path.join(img_path, '*'))
    for p in imgs:
        # 图像处理：降噪、灰度化、修改尺寸以及灰度反转
        pic = image_prepare(p)
        # 识别图像
        evaluate(pic, p)


if __name__ == '__main__':
    main()
