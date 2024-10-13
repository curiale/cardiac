"""
File: custom_objects.py
Author:Ariel Hernán Curiale and Lucca Dellazopa
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from src.settings import LABEL_TISSUES


# Custom objects for CNN Segmentation Model
# Any new custom object must be defined here and added to 'custom_objects' dict
class Z_mse(Layer):

    def __init__(self, **kwargs):
        super(Z_mse, self).__init__(**kwargs)

    def call(self, args):
        z_x, z_y = args
        return K.mean(K.square(z_x - z_y), axis=-1)

    def get_config(self):
        config = super(Z_mse, self).get_config()
        return config


class Z_norm(Layer):

    def __init__(self, **kwargs):
        super(Z_norm, self).__init__(**kwargs)

    def call(self, args):
        z, ae_z = args
        return K.sqrt(K.sum(K.square(z - ae_z), axis=-1))

    def get_config(self):
        config = super(Z_norm, self).get_config()
        return config


class Z_loss(Layer):

    def __init__(self, **kwargs):
        super(Z_loss, self).__init__(**kwargs)

    def call(self, args):
        z_mean, z_log_var = args
        val = K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
                     axis=-1)
        return val

    def get_config(self):
        config = super(Z_loss, self).get_config()
        return config


class Sampling_latent_space_epsilon(Layer):

    def __init__(self, **kwargs):
        super(Sampling_latent_space_epsilon, self).__init__(**kwargs)

    def call(self, args):
        # NOTE: por algun motivo si paso mas de 2 argumentos se pierde el
        # nombre del layer lambda.
        [z_mean_z_log_var, latent_dim] = args

        z_mean = z_mean_z_log_var[0]
        z_log_var = z_mean_z_log_var[1]

        # Random perturbation
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0.,
                                  stddev=1.)
        # tf.cast(epsilon, tf.float32)

        return z_mean + K.exp(z_log_var) * epsilon

    def get_config(self):
        config = super(Sampling_latent_space_epsilon, self).get_config()
        return config


class Sampling_latent_space(Layer):

    def __init__(self, **kwargs):
        super(Sampling_latent_space, self).__init__(**kwargs)

    def call(self, args):
        # NOTE: por algun motivo si paso mas de 2 argumentos se pierde el
        # nombre del layer lambda.
        z_mean, z_log_var = args
        return z_mean + K.exp(z_log_var)

    def get_config(self):
        config = super(Sampling_latent_space, self).get_config()
        return config


class Loss_latent_space(Layer):

    def __init__(self, **kwargs):
        super(Loss_latent_space, self).__init__(**kwargs)

    def call(self, args):
        z_mean, z_log_var = args
        val = K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
                     axis=-1)
        return val

    def get_config(self):
        config = super(Loss_latent_space, self).get_config()
        return config


class Metric():

    def __init__(self, labels):
        self._labels = np.array(labels)
        self._max_label = np.max(labels)

    def unravel_labels(self, y):
        # Data are normalized
        y = y * self._max_label
        y = K.round(y)
        y2stack = []
        for l in self._labels:
            yl = K.cast(y == l, y.dtype)
            y2stack.append(yl)

        y = K.concatenate(y2stack)
        return y


class Dice(Metric):

    def __init__(self, labels):
        super().__init__(labels)
        self.__name__ = 'dice'

    def __call__(self, y_true, y_pred):
        if K.int_shape(y_pred)[-1] == 1:
            y_true = self.unravel_labels(y_true)
            y_pred = self.unravel_labels(y_pred)
        e = K.epsilon()
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersect = 2 * K.sum(y_true_f * y_pred_f) + e
        union = K.sum(y_true_f) + K.sum(y_pred_f) + e
        return intersect / union


def tanimoto(y_true, y_pred, by_sample=False):
    '''Compute the Tanimoto coefficient (TC) which is for binary sets (one
    label) the same as the Jaccard coefficient. However, for two or more labels
    this measure represents the fuzzy variation of the Jaccard coefficient [1].

    [1] W. R. Crum et al. "Generalized overlap measures for evaluetion and
    validation in medial image analysis," IEEE Transactions on Medical Imaging,
    vol. 25, pp. 1451-1461, Nov. 2006
    '''
    axis = np.arange(K.ndim(y_true))
    if by_sample:
        axis = axis[1:]

    e = K.epsilon()
    intersect = K.sum(K.minimum(y_true, y_pred), axis=axis) + e
    union = K.sum(K.maximum(y_true, y_pred), axis=axis) + e
    tc = intersect / union
    return tc


def gjaccardd(y_true, y_pred):
    '''Compute the generalized Jaccard distance which is the fuzzy variation
    of the Jaccard distance and it should be used instead of the Jaccard
    distance when the classification has more than one class
    '''
    tc = tanimoto(y_true, y_pred)
    jd = 1 - tc
    return jd


def tanimotod(y_true, y_pred):
    tc = tanimoto(y_true, y_pred)
    td = -K.log(tc + K.epsilon())
    return td


def fake_mse(y_true, y_pred):
    return K.mean(K.square(y_pred), axis=-1)


def z_loss_mean(y_true, y_pred):
    return K.mean(y_pred)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 2. * intersection / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                K.epsilon())


def jaccard_distance(y_true, y_pred):
    '''Calculates the Jaccard index
    between predicted and target values'''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    j = intersection / (denom + K.epsilon())
    return 1 - j


# Custom objects dict
c_o = {
    'Z_mse': Z_mse,
    'Z_norm': Z_norm,
    'Z_loss': Z_loss,
    'Sampling_latent_space_epsilon': Sampling_latent_space_epsilon,
    'Sampling_latent_space': Sampling_latent_space,
    'Loss_latent_space': Loss_latent_space,
    'dice': Dice(list(LABEL_TISSUES.values())),
    'z_loss_mean': z_loss_mean,
    'tanimoto': tanimoto,
    'gjaccardd': gjaccardd,
    'jaccard_distance': jaccard_distance,
    'dice_coef': dice_coef
}
