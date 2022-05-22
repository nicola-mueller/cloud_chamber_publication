#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from glob import glob
import os

# Training script for the Attention U-Net with transfer learning
# Attention gates are added to the skip connections between encoder and decoder layers
# Allows the model to learn to only send relevant spatial information from encoder to decoder
# Paper: https://arxiv.org/abs/1804.03999

# in every tuple the first layer is the pretrained layer from the autoencoder and the second layer is the
# corresponding untrained layer in the attention u-net
layer_pairs = [("conv2d", "conv2d"),
               ("batch_normalization", "batch_normalization"),
               ("conv2d_1", "conv2d_1"),
               ("batch_normalization_1", "batch_normalization_1"),
               ("conv2d_2", "conv2d_2"),
               ("batch_normalization_2", "batch_normalization_2"),
               ("conv2d_3", "conv2d_3"),
               ("batch_normalization_3", "batch_normalization_3"),
               ("conv2d_4", "conv2d_4"),
               ("batch_normalization_4", "batch_normalization_4"),
               ("conv2d_5", "conv2d_5"),
               ("batch_normalization_5", "batch_normalization_5"),
               ("conv2d_6", "conv2d_6"),
               ("batch_normalization_6", "batch_normalization_6"),
               ("conv2d_7", "conv2d_7"),
               ("batch_normalization_7", "batch_normalization_7"),
               ("conv2d_8", "conv2d_8"),
               ("batch_normalization_8", "batch_normalization_8"),
               ("conv2d_9", "conv2d_9"),
               ("batch_normalization_9", "batch_normalization_9"),
               ("up_sampling2d", "up_sampling2d"),
               ("conv2d_10", "conv2d_10"),
               ("batch_normalization_10", "batch_normalization_12"),
               ("conv2d_12", "conv2d_17"),
               ("batch_normalization_11", "batch_normalization_13"),
               ("up_sampling2d_1", "up_sampling2d_2"),
               ("conv2d_13", "conv2d_18"),
               ("batch_normalization_12", "batch_normalization_16"),
               ("conv2d_15", "conv2d_25"),
               ("batch_normalization_13", "batch_normalization_17"),
               ("up_sampling2d_2", "up_sampling2d_4"),
               ("conv2d_16", "conv2d_26"),
               ("batch_normalization_14", "batch_normalization_20"),
               ("conv2d_18", "conv2d_33"),
               ("batch_normalization_15", "batch_normalization_21"),
               ("up_sampling2d_3", "up_sampling2d_6"),
               ("conv2d_19", "conv2d_34"),
               ("batch_normalization_16", "batch_normalization_24"),
               ("conv2d_21", "conv2d_41"),
               ("batch_normalization_17", "batch_normalization_25")
               ]

# this list is needed for freezing the encoder layers
encoder_layers = ["conv2d",
                  "batch_normalization",
                  "conv2d_1",
                  "batch_normalization_1",
                  "conv2d_2",
                  "batch_normalization_2",
                  "conv2d_3",
                  "batch_normalization_3",
                  "conv2d_4",
                  "batch_normalization_4",
                  "conv2d_5",
                  "batch_normalization_5",
                  "conv2d_6",
                  "batch_normalization_6",
                  "conv2d_7",
                  "batch_normalization_7",
                  "conv2d_8",
                  "batch_normalization_8",
                  "conv2d_9",
                  "batch_normalization_9"]
########## DATA PIPELINE ##########

# parses the compressed one hot masks
# the augmented masks are saved as one hot tensors because decompressing and compressing changes
# the RGB values of an image so they cant be saved as images
def parse_mask(name):
    mask = np.load(name)['arr_0']  # np load returns a dictionary containing the arrays that were compressed
    mask = tf.cast(mask, tf.float32)

    return mask


# parses the already cropped images
def parse_image(name):
    image = tf.io.read_file(name)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Neural Nets work with float 32

    return image


# this generator streams the images and masks to the GPU one after another
# it gets a list of pairs that correspond to the directory paths of images and their corresponding masks
# based on https://www.kaggle.com/mukulkr/camvid-segmentation-using-unet
class DataGenerator(Sequence):

    def __init__(self, pair, batch_size, shuffle):
        self.pair = pair
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    # returns the length of the data set
    def __len__(self):
        return int(tf.math.floor(len(self.pair) / self.batch_size))

    # returns a single batch
    def __getitem__(self, index):
        # a list that has the indexes of the pairs from which we want to generate images and masks for the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [k for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    # resets the pair indexes after each epoch and shuffles the indexes so that the batches
    # are in different order for every epoch
    def on_epoch_end(self):
        self.indexes = tf.range(len(self.pair))
        if self.shuffle == True:
            tf.random.shuffle(self.indexes)

    # generates a batch
    def __data_generation(self, list_IDs_temp):
        batch_images = []
        batch_masks = []

        for i in list_IDs_temp:
            image1 = parse_image(self.pair[i][0])
            batch_images.append(image1)

            mask1 = parse_mask(self.pair[i][1])
            batch_masks.append(mask1)

        # stack the images and masks of the batch into two tensors
        return tf.stack(batch_images), tf.stack(batch_masks)


# takes a path to a directory with two sub folders for training images and masks
# and returns a list of pairs of paths for images and the corresponding masks
def make_pairs(path, mode):
    pairs = []
    # sorted is important since os.path.join somehow shuffles the paths and we need
    # the image and mask paths to have the exact same order
    if mode == "train":
        image_paths = sorted(glob(os.path.join(path, "training_images/*")))
        mask_paths = sorted(glob(os.path.join(path, "training_masks/*")))

        for i in range(len(image_paths)):
            pairs.append((image_paths[i], mask_paths[i]))

        return pairs
    elif mode == "val":
        image_paths = sorted(glob(os.path.join(path, "validation_images/*")))
        mask_paths = sorted(glob(os.path.join(path, "validation_masks/*")))

        for i in range(len(image_paths)):
            pairs.append((image_paths[i], mask_paths[i]))

        return pairs
    elif mode == "test":
        image_paths = sorted(glob(os.path.join(path, "testing_images/*")))
        mask_paths = sorted(glob(os.path.join(path, "testing_masks/*")))

        for i in range(len(image_paths)):
            pairs.append((image_paths[i], mask_paths[i]))

        return pairs
    else:
        raise ValueError("invalid mode")


########## LOSS FUNCTION ##########
# based on https://github.com/aruns2120/Semantic-Segmentation-Severstal/blob/master/U-Net/CS2_firstCut.ipynb


# the dice coefficient calculates how much the predicted mask and the correct mask overlap
def dice_coef(y_true, y_predict, smooth=1):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_predict)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + smooth)


def dice_loss(y_true, y_predict):
    return (1 - dice_coef(y_true, y_predict))


# weighted variant of pixelwise_crossentropy
# based on https://www.gitmemory.com/issue/keras-team/keras/6261/569715992
def pixelwise_crossentropy(y_true, y_predicted):  #
    # weights that scale the error for each class such that they all have equal impact on the loss
    # important since the data set is very unbalanced
    # weights represent the inverse of the proportion of pixels corresponding to that class in the whole data set
    # needs to be divided by 100.0 to keep the error at a similar magnitude during training
    weight_alpha = 98.0 / 100.0
    weight_electron = 75.0 / 100.0
    weight_proton = 130.0 / 100.0
    weight_V = 263.0 / 100.0

    weights = [weight_alpha, weight_electron, weight_proton, weight_V]

    # predicted values get scaled such that they are never exactly 0 or 1 since then the logarithm diverges
    y_predicted /= tf.keras.backend.sum(y_predicted, axis=-1, keepdims=True)
    y_predicted = tf.keras.backend.clip(y_predicted,
                                        tf.keras.backend.epsilon(),
                                        1 - tf.keras.backend.epsilon())
    # compute the weighted cross_entropy
    loss = y_true * tf.keras.backend.log(y_predicted)
    loss = -tf.keras.backend.sum(loss * weights, -1)
    return loss


# defines the custom loss function, sum of dice_loss and pixelwise_crossentropy
def pce_dice_loss(y_true, y_predict):
    return pixelwise_crossentropy(y_true, y_predict) + dice_loss(y_true, y_predict)


########## NEURAL NETWORK ##########
# based on https://github.com/aruns2120/Semantic-Segmentation-Severstal/blob/master/U-Net/CS2_firstCut.ipynb


# defines the single convolutional blocks
def conv_block(input, amount_filters, kernel_size):
    x = tf.keras.layers.Conv2D(filters=amount_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal", padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters=amount_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x

# helper function for the attention block
# gets a tensor and returns a tensor consisting of rep many copies of the input tensor
# tensor of shape (x , y, z ) -> tensor of shape (x, y, rep*z)
def expand_as(tensor, rep):
    return tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3),
                                  arguments={'repnum': rep})(tensor)

# takes the output of a lower decoder layer and transforms it to have the same amount of filters as the output
# from the higher encoder layer by using a 1x1 filter
def gating_signal(input, out_size):
    x = tf.keras.layers.Conv2D(filters=out_size, kernel_size=(1, 1), padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

# code taken from https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
# and https://gist.github.com/robinvvinod/09c129b1828216da95a0d994482593ea
def attention_block(x, gating, inter_shape):
    # shapes of encoder and decoder outputs as tuples
    shape_x = tf.keras.backend.int_shape(x)
    shape_g = tf.keras.backend.int_shape(gating)

    # the output from the encoder goes through a convolution with a 2x2 stride so that the width and height is halved
    # this then matches the width and height of the gating signal
    theta_x = tf.keras.layers.Conv2D(filters=inter_shape, kernel_size=(1, 1), strides=(2, 2),
                                     padding="same")(x)
    shape_theta_x = tf.keras.backend.int_shape(theta_x)

    # this part again makes sure that the gating signal has the same dimensions as theta x
    # i'm not sure if this is necessary
    phi_g = tf.keras.layers.Conv2D(filters=inter_shape, kernel_size=(1, 1), padding="same")(gating)
    upsample_g = tf.keras.layers.Conv2DTranspose(filters=inter_shape, kernel_size=(3, 3),
                                                 strides=(shape_theta_x[1]//shape_g[1], shape_theta_x[2]//shape_g[2]),
                                                 padding="same")(phi_g)

    # add the gating signal and x
    concat_xg = tf.keras.layers.add([upsample_g, theta_x])
    activation_xg = tf.keras.layers.Activation("relu")(concat_xg)

    # computes a map of coefficients that scales each pixel of x according to the attention
    psi = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same")(activation_xg)
    sigmoid_psi = tf.keras.layers.Activation("sigmoid")(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_psi)

    # repeat psi so that every channel of x is multiplied by the attention coefficients map
    upsample_psi = tf.keras.layers.UpSampling2D(size=(shape_x[1]//shape_sigmoid[1], shape_x[2]//shape_sigmoid[2]))(sigmoid_psi)

    upsample_psi = expand_as(upsample_psi, shape_x[3])

    y = tf.keras.layers.multiply([upsample_psi, x])

    result = tf.keras.layers.Conv2D(filters=shape_x[3], kernel_size=(1, 1), padding="same")(y)
    result = tf.keras.layers.BatchNormalization()(result)
    return result


# defines the U-Net architecture
# amount filters controls the amount of filters in the convolutional layers, needs to be a power of 2!
def unet(input, amount_filters):
    # Encoder
    conv_block1 = conv_block(input, amount_filters, 3)
    pooling1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block1)
    # randomly deactivates in each training step 20% of neurons, gives better generalization
    dropout1 = tf.keras.layers.Dropout(0.2)(pooling1)

    conv_block2 = conv_block(dropout1, amount_filters * 2, 3)
    pooling2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block2)
    dropout2 = tf.keras.layers.Dropout(0.2)(pooling2)

    conv_block3 = conv_block(dropout2, amount_filters * 4, 3)
    pooling3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block3)
    dropout3 = tf.keras.layers.Dropout(0.2)(pooling3)

    conv_block4 = conv_block(dropout3, amount_filters * 8, 3)
    pooling4 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block4)
    dropout4 = tf.keras.layers.Dropout(0.2)(pooling4)

    encoded_features = conv_block(dropout4, amount_filters * 16, 3)

    # Decoder
    upsample_block1 = tf.keras.layers.UpSampling2D()(encoded_features)
    upsample_block1 = tf.keras.layers.Conv2D(filters=amount_filters * 8, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block1)
    gating1 = gating_signal(encoded_features, amount_filters * 8)
    attention1 = attention_block(conv_block4, gating1, amount_filters * 8)
    upsample_block1 = tf.keras.layers.concatenate([upsample_block1, attention1])  # skip connection
    dropout5 = tf.keras.layers.Dropout(0.2)(upsample_block1)
    conv_block5 = conv_block(dropout5, amount_filters * 8, 3)

    upsample_block2 = tf.keras.layers.UpSampling2D()(conv_block5)
    upsample_block2 = tf.keras.layers.Conv2D(filters=amount_filters * 4, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block2)
    gating2 = gating_signal(conv_block5, amount_filters * 4)
    attention2 = attention_block(conv_block3, gating2, amount_filters * 4)
    upsample_block2 = tf.keras.layers.concatenate([upsample_block2, attention2])
    dropout6 = tf.keras.layers.Dropout(0.2)(upsample_block2)
    conv_block6 = conv_block(dropout6, amount_filters * 4, 3)

    upsample_block3 = tf.keras.layers.UpSampling2D()(conv_block6)
    upsample_block3 = tf.keras.layers.Conv2D(filters=amount_filters * 2, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block3)
    gating3 = gating_signal(conv_block6, amount_filters * 2)
    attention3 = attention_block(conv_block2, gating3, amount_filters * 2)
    upsample_block3 = tf.keras.layers.concatenate([upsample_block3, attention3])
    dropout7 = tf.keras.layers.Dropout(0.2)(upsample_block3)
    conv_block7 = conv_block(dropout7, amount_filters * 2, 3)

    upsample_block4 = tf.keras.layers.UpSampling2D()(conv_block7)
    upsample_block4 = tf.keras.layers.Conv2D(filters=amount_filters, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block4)
    gating4 = gating_signal(conv_block7, amount_filters)
    attention4 = attention_block(conv_block1, gating4, amount_filters)
    upsample_block4 = tf.keras.layers.concatenate([upsample_block4, attention4])
    dropout8 = tf.keras.layers.Dropout(0.2)(upsample_block4)
    conv_block8 = conv_block(dropout8, amount_filters, 3)

    # amount of filters in output layer needs to be equal to the amount of classes
    output = tf.keras.layers.Conv2D(filters=5, kernel_size=(1, 1), activation="sigmoid")(conv_block8)
    unet = tf.keras.Model(inputs=[input], outputs=[output])
    return unet


########## TRAINING ##########

# this creates the data generator that is given to the neural net
train_pairs = make_pairs("", mode="train")
batch_size = 8
trainset_length = len(train_pairs)
steps_per_epoch = trainset_length // batch_size

train_generator = DataGenerator(pair=train_pairs,
                                batch_size=batch_size, shuffle=True)

val_pairs = make_pairs("", mode="val")

val_generator = DataGenerator(pair=val_pairs,
                              batch_size=batch_size, shuffle=False)


# define a mirror strategy object that makes copies of the network run on multiple GPU's
# the gradients of each copy are combined
# the network needs to be defined inside the scope of the mirrored strategy object
# https://www.tensorflow.org/guide/distributed_training
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    input = tf.keras.layers.Input(shape=[992, 1312, 3])  # shape of the input images
    unet = unet(input, 8)

    # compiles the neural net
    optimizer = tf.keras.optimizers.Adam()  # better than stochastic gradient decent
    unet.compile(optimizer=optimizer, loss=pce_dice_loss, metrics=[dice_coef, pixelwise_crossentropy])

    # load pretraining.h5
    pretrained_unet = tf.keras.models.load_model("")
    pretrained_layers = pretrained_unet.layers

    # copies the pre-trained weights
    for names in layer_pairs:
        untrained_layer = unet.get_layer(name=names[1])  # retrieves the untrained layer
        pretrained_layer = pretrained_unet.get_layer(name=names[0])  # retrieves the trained layer
        untrained_layer.set_weights(pretrained_layer.get_weights())  # copies weights

    # freezes the encoder layers
    for name in encoder_layers:
        layer = unet.get_layer(name=name)
        layer.trainable = False


# callback for logging training metrics that can be displayed in Tensorboard
# with the command 'Tensorboard --logdir training_logs/train'
metrics_logger = tf.keras.callbacks.TensorBoard(log_dir="", update_freq='epoch', write_images=True)

# callback for saving the best model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("", monitor='val_loss', save_best_only=True)

# stop training when no progress is made
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)

# trains it
unet_history = unet.fit(x=train_generator,
                        validation_data=val_generator,
                        epochs=80,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[model_checkpoint])

# saves the computed weights and the network architecture
# when loading the neural net from the h5 file it first needs to be recompiled
# since tensorflow has trouble with the custom loss function
unet.save("attention.h5")
