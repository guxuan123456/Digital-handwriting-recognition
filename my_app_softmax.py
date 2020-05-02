# coding=utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data


'''
Tensorflow 依赖于一个高效的 C++ 后端来进行计算.与后端的这个连接叫做 session.一般而言,使用 TensorFlow 程序的流程是先创建一个图,然后在 session 中启动它.
'''
sess = tf.Session()

#加载MNIST数据
# 下载MNIST数据集到'MNIST_data'文件夹并解压
mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
#这里,mnist是一个轻量级的类.它以 Numpy 数组的形式存储着训练、校验和测试数据集.同时提供了一个函数,用于在迭代中获得 minibatch,后面我们将会用到.

# 设置权重weights和偏置biases作为优化变量，初始值设为0
# 机器学习中，模型参数一般用Variable来表示。
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
'''
我们在调用tf.Variable的时候传入初始值.在这个例子里,我们把W和b都初始化为零向量.W是一个784x10的矩阵(因为我们有784个特征和10个输出值).b是一个10维的向量(因为我们有10个分类).
变量需要通过 seesion 初始化后,才能在session中使用.这一初始化步骤为,为初始值指定具体值(本例当中是全为零),并将其分配给每个变量,可以一次性为所有变量完成此操作.
'''
init = tf.initialize_all_variables()
sess.run(init)

# 构建模型,建立一个拥有一个线性层的 softmax 回归模型
'''
这里的x和y并不是特定的值,相反,他们都只是一个占位符,可以在 TensorFlow 运行某一计算时根据该占位符输入具体的值.
输入图片x是一个二维的浮点数张量.这里,分配给它的shape为[None, 784],其中784是一张展平的 MNIST 图片的维度.None表示其值大小不定,在这里作为第一个维度值,用以指代batch的大小,意即x的数量不定.输出类别值y_也是一个二维张量,其中每一行为一个10维的 one-hot 向量,用于代表对应某一 MNIST 图片的类别.
虽然placeholder的shape参数是可选的,但有了它,TensorFlow 能够自动捕捉因数据维度不一致导致的错误.
'''
x = tf.placeholder("float", [None, 784])
y = tf.nn.softmax(tf.matmul(x, weights) + biases)                                   # 模型的预测值，两个矩阵相乘
y_real = tf.placeholder("float", [None, 10])                                        # 真实值


# 损失函数为目标类别和预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_real * tf.log(y))                                  # 预测值与真实值的交叉熵
#注意,tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了.我们计算的交叉熵是指整个minibatch的.

# 最快速下降法，步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)        # 使用梯度下降优化器最小化交叉熵
'''
这一行代码实际上是用来往计算图上添加一个新操作,其中包括计算梯度,计算每个参数的步长变化,并且计算出新的参数值.
返回的train_step操作对象,在运行时会使用梯度下降来更新参数.因此,整个模型的训练可以通过反复地运行train_step来完成.
'''
'''
 上面是将各种要用的数据都定义了，感觉像是定义函数一样，因为前面的
 x,y吧位置占了，下面就是从数据集中读取数据然后将数据对应到前面的x,y就可以运行得到模型。
'''
# 开始训练
for i in range(1000):
    # 每一步迭代会加载50个训练样本，然后执行一次train_step，，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代.
    # 可以用feed_dict代替任何张量.
    batch_xs, batch_ys = mnist.train.next_batch(100)                                # 每次随机选取100个数据进行训练，即所谓的“随机梯度下降（Stochastic Gradient Descent，SGD）”
    sess.run(train_step, feed_dict={x: batch_xs, y_real:batch_ys})                  # 正式执行train_step，用feed_dict的数据取代placeholder

    # 评估模型的性能
    if i % 100 == 0:
        # 每训练100次后评估模型

        '''
        首先让我们找出那些预测正确的标签.tf.argmax 是一个非常有用的函数,它能给出某个 tensor 对象在某一维上的其数据最大值所在的索引值.由于标签向量是由0,1组成,因此最大值1所在的索引位置就是类别标签,比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值,而 tf.argmax(y_,1) 代表正确的标签,我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配).
        '''
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_real, 1))       # 比较预测值和真实值是否一致
        
        '''
        这里返回一个布尔数组.为了计算我们分类的准确率,我们将布尔值转换为浮点数来代表对、错,然后取平均值.例如：[True, False, True, True]变为[1,0,1,1],计算出平均值为0.75.
        '''
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))             # 统计预测正确的个数，取均值得到准确率
        #print(accuracy)
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels}))#这一行为啥这样我也不知道.


'''
这是最简单的一种手写体识别了，接着我要做个超级复杂的。
准备采用卷积神经网络使训练精度提高，上面这个仅仅使用了tf中的softmax分类器。所以导致正确率只有91%，后面通过卷积神经网络可以提高识别率，准确率达到99%左右。
'''