from libraries import *
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from sklearn.extmath import cartesian
from medpy.metric.binary import __surface_distances

# https://github.com/danielenricocahall/Keras-Weighted-Hausdorff-Distance-Loss
def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def weighted_hausdorff_distance(w, h, alpha):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss

# https://github.com/JunMa11/AbdomenCT-1K/blob/main/1-FullySupervisedLearning/nnUNet/nnunet/evaluation/surface_dice.py
def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy
    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)

    numel_a = len(a_to_b)
    numel_b = len(b_to_a)

    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b

    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b

    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc

def dice_coef(y_true, y_pred, smooth:int=100):
    # y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=class_nums+1)[...,1:])
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred[...])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    dice = K.mean((2. * intersect / (denom + smooth)))
    return dice