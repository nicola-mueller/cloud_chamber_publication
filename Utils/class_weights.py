import tensorflow as tf
from glob import glob
import os
import numpy as np
# turn off GPU processing because
# tensorflow-gpu can lead to trouble if not installed correctly
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# This script computes the loss weights for every class.
# These weights balance the loss and thus make the neural network focus
# on rare and common particles equally much.


# parses the masks
def parse_mask(name):
    # print(name)
    mask = np.load(name)['arr_0']

    return mask

def create_mask(one_hot):
    # input is a tensor that contains the class probability maps

    # colors for the predicted mask, need to be divided by 255.0 since we work with float colors
    colors = [
        [0.0 / 255.0, 100.0 / 255.0, 0.0 / 255.0],  # Green for Alpha particles
        [0.0 / 255.0, 0.0 / 255.0, 205.0 / 255.0],  # Blue for electrons
        [227.0 / 255.0, 26.0 / 255.0, 28.0 / 255.0],  # Red for Protons
        [139.0 / 255.0, 0.0 / 255.0, 139.0 / 255.0],  # Violet for Alpha_V particles
        [0.0, 0.0, 0.0]  # Black
    ]

    # unstack to get a list of class probability maps, where each cell has the predicted probability
    # that the corresponding pixel has the color of the class
    raw_channels = tf.unstack(one_hot, axis=-1)

    # apply thresholds, every probability lower than 50% is set to 0 since we only want confident predictions
    # changing the thresholds changes the final masks a lot
    threshold_channel0 = tf.math.maximum(raw_channels[0] - 0.9, 0)  # Alpha
    threshold_channel1 = tf.math.maximum(raw_channels[1] - 0.9, 0)  # Electron
    threshold_channel2 = tf.math.maximum(raw_channels[2] - 0.9, 0)  # Proton
    threshold_channel3 = tf.math.maximum(raw_channels[3] - 0.9, 0)  # V
    threshold_channel4 = tf.math.maximum(raw_channels[4] - 0.9, 0)  # Black


    # stack the class maps with the thresholds applied to a tensor so that
    # we can later apply the arg max function
    threshold_one_hot = tf.stack([threshold_channel0, threshold_channel1,
                                  threshold_channel2, threshold_channel3,
                                  threshold_channel4], axis=-1)

    # green, blue, red, violet, black
    class_indices = [0, 1, 2, 3, 4]
    # returns a 2d class map where in every pixel stands the class index of the
    # class that had the highest probability for that pixel
    argmax_mask = tf.argmax(one_hot, axis=-1)

    # now for every class index we create a 0-1 encoded (one hot) class map
    # so that when we later multiply by the RGB color we get for each pixel either black
    # or the correct color
    channels = []
    for index in class_indices:
        # returns one hot class map where a pixel has value 1 if the corresponding
        # class was the one with highest probability for that pixel
        class_map = tf.equal(argmax_mask, index)
        class_map = tf.cast(class_map, tf.float32)
        channels.append(class_map)

    # stack 3 copies of a one hot class map together and multiply by the corresponding
    # RGB color to get an RGB class map
    red = tf.stack([channels[0], channels[0], channels[0]], axis=-1) * colors[0]
    green = tf.stack([channels[1], channels[1], channels[1]], axis=-1) * colors[1]
    violet = tf.stack([channels[2], channels[2], channels[2]], axis=-1) * colors[2]
    blue = tf.stack([channels[3], channels[3], channels[3]], axis=-1) * colors[3]
    black = tf.stack([channels[4], channels[4], channels[4]], axis=-1) * colors[4]

    # add all the RGB class maps together to get the final mask
    mask = tf.math.add(red, green)
    mask = tf.math.add(mask, violet)
    mask = tf.math.add(mask, blue)
    mask = tf.math.add(mask, black)

    return mask

# gets the paths of the masks in a given directory
def get_mask_paths(path):
    mask_paths = sorted(glob(os.path.join(path, "validation_masks/*")))
    return mask_paths

mask_paths = get_mask_paths("/DataSet/validation")

colors = [
        [0.0 / 255.0, 100.0 / 255.0, 0.0 / 255.0],  # Green for Alpha particles
        [0.0 / 255.0, 0.0 / 255.0, 205.0 / 255.0],  # Blue for electrons
        [227.0 / 255.0, 26.0 / 255.0, 28.0 / 255.0],  # Red for Protons
        [139.0 / 255.0, 0.0 / 255.0, 139.0 / 255.0],  # Violet for Alpha_V particles
        [0.0, 0.0, 0.0]  # Background
    ]


# initializes the counter for the pixels of the different colors
color_pixels = [0, 0, 0, 0, 0]

total_pixels = 992 * 1312 * len(mask_paths)

# parses each mask individually and computes the number of pixels of every color
mask_counter = 0
for mask_path in mask_paths:
    mask = parse_mask(mask_path)
    mask = create_mask(mask)
    mask_counter = mask_counter + 1
    print(mask_counter)
    # looks at each color individually
    for color_index in range(len(colors)):
        # tf.equal compares every pixel of the mask with the current RGB color
        # and returns a matrix where a cell is TRUE if the corresponding pixel had the color of the current class
        # reduce all then turns this boolean matrix into a 2D map
        color_mask = tf.reduce_all(tf.equal(mask, colors[color_index]), axis=-1)
        # casts the boolean color map to integers and the sums all cells with value 1 up
        color_pixel_count = tf.reduce_sum(tf.cast(color_mask, tf.int64))
        # updates the pixel counter for the current class
        color_pixels[color_index] = color_pixels[color_index] + color_pixel_count

# computes the class weights as the inverse of the proportion that each color had among all pixels
green_weight = (1 / color_pixels[0]) * total_pixels
blue_weight = (1 / color_pixels[1]) * total_pixels
red_weight = (1 / color_pixels[2]) * total_pixels
violet_weight = (1 / color_pixels[3]) * total_pixels
black_weight = (1 / color_pixels[4]) * total_pixels

print(str(mask_counter) + " masks analyzed.")
print("weight for alphas: " + str(green_weight))
print("weight for electrons: " + str(blue_weight))
print("weight for protons: " + str(red_weight))
print("weight for Vs: " + str(violet_weight))
print("weight for background: " + str(black_weight))
