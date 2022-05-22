import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### UTILS ###

# parses an image from the given path and converts the data type
def parse_image(name):
    image = tf.io.read_file(name)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# turns a prediction into a mask
def create_mask(one_hot):
    # input is a tensor that contains the class probability maps

    # colors for the predicted mask, need to be divided by 255.0 since we work with float colors
    colors = [
        # [0.0 / 255.0, 100.0 / 255.0, 0.0 / 255.0],  # Green for Alpha particles
        [200.0 / 255.0, 200.0 / 255.0, 0.0 / 255.0],  # Yellow for Alpha particles
        [0.0 / 255.0, 0.0 / 255.0, 205.0 / 255.0],  # Blue for electrons
        [227.0 / 255.0, 26.0 / 255.0, 28.0 / 255.0],  # Red for Protons
        [139.0 / 255.0, 0.0 / 255.0, 139.0 / 255.0],  # Violet for Alpha_V particles
        [0.0, 0.0, 0.0]  # Black
    ]

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
    green = tf.stack([channels[0], channels[0], channels[0]], axis=-1) * colors[0]
    blue = tf.stack([channels[1], channels[1], channels[1]], axis=-1) * colors[1]
    red = tf.stack([channels[2], channels[2], channels[2]], axis=-1) * colors[2]
    violet = tf.stack([channels[3], channels[3], channels[3]], axis=-1) * colors[3]
    black = tf.stack([channels[4], channels[4], channels[4]], axis=-1) * colors[4]

    # add all the RGB class maps together to get the final mask
    mask = tf.math.add(red, green)
    mask = tf.math.add(mask, violet)
    mask = tf.math.add(mask, blue)
    mask = tf.math.add(mask, black)

    return mask

# returns for a prediction a binary image containing the particles of the specified color
def make_binary(prediction, color):
    # black, red, green, violet, blue
    class_indices = [0, 1, 2, 3, 4]
    # returns a 2d class map where in every pixel stands the class index of the
    # class that had the highest probability for that pixel
    argmax_mask = tf.argmax(prediction, axis=-1)

    # now for every class index we create a 0-1 encoded (one hot) class map
    channels = []
    for index in class_indices:
        # returns one hot class map where a pixel has value 1 if the corresponding
        # class was the one with highest probability for that pixel
        class_map = tf.equal(argmax_mask, index)
        class_map = tf.cast(class_map, tf.uint8)
        channels.append(class_map)

    if color == "red":
        binary = channels[2]
        return binary
    elif color == "violet":
        binary = channels[3]
        return binary
    elif color == "blue":
        binary = channels[1]
        return binary
    elif color == "green":
        binary = channels[0]
        return binary
    elif color == "green_red_violet":
        binary = np.add(channels[0], channels[2])
        binary = np.add(binary, channels[3])
        return binary
    else:
        raise ValueError("wrong color")
