# Pytorch中的5种normalization方法
   ##什么是数据归一化？
    在数据中往往不同指标上存在不同的量纲和量纲单位，会影响最终的分析结果，为了消除指标之间的量纲影响，需要对数据进行标准化，以解决数据指标之间的可比性。常用的方法有【0， 1】归一化和正态分布归一化。
   ##为什么需要进行normalization？
        往往只对输入数据进行归一化，当数据在网络中传播时，后面每一层的输入数据分布是一直发生变化的，不利于网络的训练。
        采用normalization的好处有1.加速网络模型的收敛速度，2.缓解深层网络中的‘梯度弥散’，使得深层网络的训练变得容易和稳定。
## Batch normalization
    定义：通道级别的归一化，对于整个mini batch中每个通道进行归一化，【batch， c， timesteps】 -> 【c】， 【batch， c， h， w】 -> 【c】
    1.训练和推理时的参数不一致，采用的均值和方差不相同，训练时采用的是当前输入mini batch的均值和方差，推理时采用的全体训练数据的经过滑动平均的均值和方差，
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    2.每个batch不能太小，小的batch效果不好
    3.batchnormalization不适用于NLP任务中的原因在于：会对timesteps维度上求平均值，而NLP任务中每条数据的timesteps往往是不相同的，需要补0，求平均会引入误差，
    
## Layer normalization
    定义：每一时刻的归一化，只对c求均值、标准差，常常用于NLP任务上，【batch， c， timesteps】 -> 【batch， timesteps】， 【batch， c， h， w】 -> 【batch， h， w】
    
## Instance normalization
    定义：每一个样本的归一化，只对timesteps求均值、方差，【batch， c， timesteps】 -> 【batch， c】， 【batch， c， h， w】 -> 【batch， c】
    为什么能够实现风格迁移？ 因为对于timesteps进行归一化，会消去数据在timesteps上不变的东西，在视频上是风格，在语音上是身份
    
## Group normalization
    定义：将c通道划分为n个group， 每个group中包含k个通道，n * k = c， 对每个组中的timesteps和c求均值和标准差，【batch， c//num_group， timesteps】 -> 【batch， num_group】，
    【batch， c//num_group， h， w】 -> 【batch， num_group】
    
## Weight normalization
    定义：对于一个向量，保留它的方向，并给它增添一个可学习的幅度。