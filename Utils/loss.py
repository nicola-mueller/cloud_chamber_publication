import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### LOSS ###

def dice_coef(y_true, y_predict, smooth=1):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_predict)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + smooth)


def dice_loss(y_true, y_predict):
    return (1 - dice_coef(y_true, y_predict))


def pixelwise_crossentropy(y_true, y_predicted):
    weight_alpha = 99.8 / 100.0
    weight_electron = 76.0 / 100.0
    weight_proton = 134.3 / 100.0
    weight_V = 275.2 / 100.0
    weight_background = 10.0 / 100.0

    weights = [weight_alpha, weight_electron, weight_proton, weight_V, weight_background]
    y_predicted /= tf.keras.backend.sum(y_predicted, axis=-1, keepdims=True)
    y_predicted = tf.keras.backend.clip(y_predicted,
                                        tf.keras.backend.epsilon(),
                                        1 - tf.keras.backend.epsilon())
    loss = y_true * tf.keras.backend.log(y_predicted)
    loss = -tf.keras.backend.sum(loss * weights, -1)
    return loss


def pce_dice_loss(y_true, y_predict):
    return pixelwise_crossentropy(y_true, y_predict) + dice_loss(y_true, y_predict)
