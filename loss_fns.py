from libraries import *
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

# custom spatial softmax
# https://github.com/tensorflow/tensorflow/issues/6271#issuecomment-266893850
class SpatialSoftmax(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(SpatialSoftmax, self).__init__(**kwargs)
    
    def call(self, inputs):
        N = inputs.shape[0]
        H = inputs.shape[1]
        W = inputs.shape[2]
        inputs = tf.reshape(inputs, [H * W])
        softmax = tf.nn.softmax(inputs)
        return softmax
# attention loss
# https://github.com/InhwanBae/ENet-SAD_Pytorch/blob/master/model_ENET_SAD.py
def at_loss(x1, x2):
    """
    x1 - previous step feature map
    x2 - current step feature map
    """
    # G^2_sum
    sps = SpatialSoftmax()
    if x1.shape != x2.shape:
        x1 = tf.reduce_sum(tf.pow(x1, 2), 2)
        x1 = sps(tf.expand_dims(x1, 0))
        x2 = tf.reduce_sum(tf.pow(x2, 2), 2)
        x2 = tf.squeeze(UpSampling2D(size = 2, interpolation='bilinear')(tf.expand_dims(tf.expand_dims(x2, axis=0), axis=-1)), axis=-1)
        x2 = sps(x2)
    else:
        x1 = tf.reduce_sum(tf.pow(x1, 2), 2)
        x1 = sps(x1)
        x2 = tf.reduce_sum(tf.pow(x2, 2), 2)
        x2 = sps(x2)

    loss = tf.keras.losses.MeanSquaredError()(x1, x2)
    return loss

# dice loss
# https://github.com/keras-team/keras/issues/9395#issuecomment-370971561
def dice_coef(y_true, y_pred, smooth:int=100):
    # y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=class_nums+1)[...,1:])
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred[...])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    dice = 1 - K.mean((2. * intersect / (denom + smooth)))
    return dice
    
# computes model loss for training
def dice_loss_fn(y_true, y_pred, sa:bool=True):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    total_loss = []
    if sa == True:
        for i in range(batch_len):
            output = y_pred[0][i, :, :, :]
            x1 = y_pred[1][i, :, :, :]
            x2 = y_pred[2][i, :, :, :]
            x3 = y_pred[3][i, :, :, :]
            x4 = y_pred[4][i, :, :, :]
            at_loss_1 = at_loss(x1, x2)
            at_loss_2 = at_loss(x2, x3)
            at_loss_3 = at_loss(x3, x4)
            at_loss_scale = 0.1
            
            total_at_loss = (at_loss_1 + at_loss_2 + at_loss_3)
            seg_loss = dice_coef(y_true[i], output)
            total_loss.append(seg_loss + at_loss_scale * total_at_loss)
    else:
        for i in range(batch_len):
            output = y_pred[0][i, :, :, :]
        
            seg_loss = dice_coef(y_true[i], output)
            total_loss.append(seg_loss)
    # return tf.reduce_mean(tf.stack(total_loss), axis=-1)
    return tf.reduce_mean(tf.stack(total_loss), axis=-1)

# weighted categorical cross entropy loss function
def wce_loss(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        y_true_w = tf.math.multiply(weights, y_true)
        loss = tf.math.multiply(y_true_w, K.log(y_pred))
        loss = -K.sum(loss, -1)
        loss = K.mean(loss)
        return loss

    return loss
# computes model loss for training
def wce_loss_fn(y_true, y_pred, weights, class_nums:int=1, sa:bool=True):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    total_loss = []
    if sa == True:
        for i in range(batch_len):
            output = y_pred[0][i, :, :, :]
            x1 = y_pred[1][i, :, :, :]
            x2 = y_pred[2][i, :, :, :]
            x3 = y_pred[3][i, :, :, :]
            x4 = y_pred[4][i, :, :, :]
            at_loss_1 = at_loss(x1, x2)
            at_loss_2 = at_loss(x2, x3)
            at_loss_3 = at_loss(x3, x4)
            at_loss_scale = 0.1
            
            total_at_loss = (at_loss_1 + at_loss_2 + at_loss_3)
        
            seg_loss = wce_loss(weights)(y_true[i], output)
            total_loss.append(seg_loss + at_loss_scale * total_at_loss)
    else:
        for i in range(batch_len):
            output = y_pred[0][i, :, :, :]
        
            seg_loss = wce_loss(weights)(y_true[i], output)
            total_loss.append(seg_loss)
    return tf.reduce_mean(tf.stack(total_loss), axis=-1)
# calculate weights for weighted cross entropy loss
def calc_loss_weights(labels_path, class_nums=2):
    # weights = np.empty((len(labels_path), class_nums+1))
    pixels_sums = np.empty((len(labels_path), class_nums+1))
    for index, label_path in enumerate(labels_path.values()):
        # weights.append([])
        loaded_label = np.load(label_path)
        for i in range(loaded_label.shape[-1]):
            label = loaded_label[..., i]
            pixels_sum = np.sum(label)
            pixels_sums[index][i] = pixels_sum
    
    final_sums = pixels_sums.sum(0)
    ratio = final_sums / (pixels_sums.shape[0] * loaded_label.shape[0] * loaded_label.shape[1])
    weights = 1 / ((ratio * class_nums) + 1e-7)

    # weights = np.mean(weights, 0)
    print(loaded_label.shape)
    print(weights)

    return weights
    