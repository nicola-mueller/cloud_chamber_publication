import numpy as np
import cv2
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### POST-PROCESSING ###

# creates for a prediction an image without the electrons and with datatype uint8
def create_mask_no_electrons(one_hot):
    # input is a tensor that contains the class probability maps

    # colors for the predicted mask, need to be divided by 255.0 since we work with float colors
    colors = [
        [0, 100, 0],  # Green for Alpha particles
        [0, 0, 205],  # Blue for electrons
        [227, 26, 28],  # Red for Protons
        [139, 0, 139],  # Violet for Alpha_V particles
        [0, 0, 0]  # Black
    ]

    # green, blue, red, violet black
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
        class_map = tf.cast(class_map, tf.uint8)
        channels.append(class_map)

    # stack 3 copies of a one hot class map together and multiply by the corresponding
    # RGB color to get an RGB class map
    green = tf.stack([channels[0], channels[0], channels[0]], axis=-1) * colors[0]
    blue = tf.stack([channels[1], channels[1], channels[1]], axis=-1) * colors[4]
    red = tf.stack([channels[2], channels[2], channels[2]], axis=-1) * colors[0]
    violet = tf.stack([channels[3], channels[3], channels[3]], axis=-1) * colors[0]
    black = tf.stack([channels[4], channels[4], channels[4]], axis=-1) * colors[4]

    # add all the RGB class maps together to get the final mask
    mask = tf.math.add(red, green)
    mask = tf.math.add(mask, violet)
    mask = tf.math.add(mask, blue)
    mask = tf.math.add(mask, black)

    return mask

###   NOT USED   ###
# colors each particle according to the color that has the most pixels in that particle
def majority_vote(prediction):
    prediction_channels = tf.unstack(prediction, axis=-1)

    # stack the predictions of the big particles and background
    big_particles = tf.stack([prediction_channels[4], prediction_channels[0],
                              prediction_channels[2], prediction_channels[3], prediction_channels[4]], axis=-1)

    # returns a mask that has in every pixel the index of the color that had the highest probability
    argmax_mask = tf.argmax(big_particles, axis=-1)

    # build an image out of the prediction for connected component analysis
    visualized_mask = create_mask_no_electrons(prediction).numpy()

    # greyscale image required for connected components analysis
    mask_bw = cv2.cvtColor(visualized_mask, cv2.COLOR_BGR2GRAY)

    connected_components = cv2.connectedComponentsWithStats(image=mask_bw, connectivity=4, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_components

    green_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    red_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    violet_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    # for every connected component we create a mask, multiply it with the arg max labels of the big particles
    # and count which color has the most pixels
    for i in range(1, numLabels):
        # returns a mask that has 1's only where the current connected component is
        componentMask = (labels == i).astype("uint8")

        component_labels = componentMask * argmax_mask

        green_count = np.count_nonzero(component_labels == 1)
        # print(green_count)
        red_count = np.count_nonzero(component_labels == 2)
        # print(red_count)
        violet_count = np.count_nonzero(component_labels == 3)
        # print(violet_count)

        max_count = max(green_count, red_count, violet_count)
        # print(max_count)

        # if any of the colors of the big particles were present in the connected component
        # we add the connected component to the mask of the corresponding color
        # values are multiplied by 10 to make sure the new color assignments have the highest probability
        if max_count > 0:
            if max_count == green_count:
                green_mask = np.add(green_mask, componentMask * 10)
            elif max_count == red_count:
                red_mask = np.add(red_mask, componentMask * 10)
            elif max_count == violet_count:
                violet_mask = np.add(violet_mask, componentMask * 10)

    green_mask = tf.cast(green_mask, tf.float32)
    red_mask = tf.cast(red_mask, tf.float32)
    violet_mask = tf.cast(violet_mask, tf.float32)

    # build the updated prediction
    majority_vote_prediction = tf.stack([green_mask, prediction_channels[1],
                                         red_mask, violet_mask,
                                         prediction_channels[4]], axis=-1)

    return majority_vote_prediction

# pixels that belong to a particle that is too small to be a proton will be assigned to the class with next higher
# probability
def filterRed(prediction, threshold):
    # get the pixels for which red has the highest probability
    class_indices = [0, 1, 2, 3, 4]
    argmax_mask = tf.argmax(prediction, axis=-1)

    oneHotChannels = []
    for index in class_indices:
        class_map = tf.equal(argmax_mask, index)
        class_map = tf.cast(class_map, tf.uint8)
        oneHotChannels.append(class_map)

    # build an image with only the red particles and turn it to grayscale
    red = tf.stack([oneHotChannels[2], oneHotChannels[2], oneHotChannels[2]], axis=-1) * [227, 26, 28]
    red_grey = cv2.cvtColor(red.numpy(), cv2.COLOR_BGR2GRAY)

    connected_components = cv2.connectedComponentsWithStats(image=red_grey, connectivity=4, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_components
    not_proton_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    # build a mask with 1's for the pixels of the areas that are too small
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < threshold:
            componentMask = (labels == i).astype("uint8")
            not_proton_mask = cv2.bitwise_or(not_proton_mask, componentMask)


    invert_not_proton_mask = (not_proton_mask == 0).astype(np.uint8)

    channels = tf.unstack(prediction, axis=-1)
    filtered_protons = channels[2] * invert_not_proton_mask

    updatedPrediction = tf.stack([channels[0], channels[1],
                                  filtered_protons, channels[3],
                                  channels[4]], axis=-1)

    return updatedPrediction

# colors each red particle with less pixels than the threshold black
def filterRedToBlack(prediction, threshold):
    # get the pixels for which red has the highest probability
    class_indices = [0, 1, 2, 3, 4]
    argmax_mask = tf.argmax(prediction, axis=-1)

    oneHotChannels = []
    for index in class_indices:
        class_map = tf.equal(argmax_mask, index)
        class_map = tf.cast(class_map, tf.uint8)
        oneHotChannels.append(class_map)

    # build an image with only the red particles and turn it to grayscale
    red = tf.stack([oneHotChannels[2], oneHotChannels[2], oneHotChannels[2]], axis=-1) * [227, 26, 28]
    red_grey = cv2.cvtColor(red.numpy(), cv2.COLOR_BGR2GRAY)

    connected_components = cv2.connectedComponentsWithStats(image=red_grey, connectivity=4, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_components
    mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    # build a mask with 1's for the pixels of the areas that are too small
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < threshold:
            componentMask = (labels == i).astype("uint8")
            mask = cv2.bitwise_or(mask, componentMask)

    predictionChannels = tf.unstack(prediction, axis=-1)
    # we add the mask of the too small pixels to the mask of the background
    # values multiplied by 10 to make sure that those pixels are assigned to black
    updatedBackground = np.add(predictionChannels[4], mask*10)

    updatedPrediction = tf.stack([predictionChannels[0], predictionChannels[1],
                                  predictionChannels[2], predictionChannels[3],
                                  updatedBackground], axis=-1)
    return updatedPrediction

# pixels that belong to a particle that is too small to be a V will be assigned to the class with next higher
# probability
def filterViolet(prediction, threshold):
    # get the pixels for which violet has the highest probability
    class_indices = [0, 1, 2, 3, 4]
    argmax_mask = tf.argmax(prediction, axis=-1)

    oneHotChannels = []
    for index in class_indices:
        class_map = tf.equal(argmax_mask, index)
        class_map = tf.cast(class_map, tf.uint8)
        oneHotChannels.append(class_map)

    # build an image with only the violet particles and turn it to grayscale
    violet = tf.stack([oneHotChannels[3], oneHotChannels[3], oneHotChannels[3]], axis=-1) * [139, 0, 139]
    violet_grey = cv2.cvtColor(violet.numpy(), cv2.COLOR_BGR2GRAY)

    connected_components = cv2.connectedComponentsWithStats(image=violet_grey, connectivity=4, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_components
    not_V_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    # build a mask with 1's for the pixels of the areas that are too small
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < threshold:
            componentMask = (labels == i).astype(np.uint8)
            not_V_mask = cv2.bitwise_or(not_V_mask, componentMask)

    invert_not_V_mask = (not_V_mask == 0).astype(np.uint8)

    channels = tf.unstack(prediction, axis=-1)
    filtered_V = channels[3] * invert_not_V_mask

    updatedPrediction = tf.stack([channels[0], channels[1],
                                  channels[2], filtered_V,
                                  channels[4]], axis=-1)

    return updatedPrediction

# colors each green particle with less pixels than the threshold black
def filterGreenToBlack(prediction, threshold):
    # get the pixels for which green has the highest probability
    class_indices = [0, 1, 2, 3, 4]
    argmax_mask = tf.argmax(prediction, axis=-1)

    oneHotChannels = []
    for index in class_indices:
        class_map = tf.equal(argmax_mask, index)
        class_map = tf.cast(class_map, tf.uint8)
        oneHotChannels.append(class_map)

    # build an image with only the green particles and turn it to grayscale
    green = tf.stack([oneHotChannels[0], oneHotChannels[0], oneHotChannels[0]], axis=-1) * [0, 100, 0]
    green_grey = cv2.cvtColor(green.numpy(), cv2.COLOR_BGR2GRAY)

    # plt.imshow(green_grey, cmap='gray')
    # plt.show()

    connected_components = cv2.connectedComponentsWithStats(image=green_grey, connectivity=4, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_components
    mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    # build a mask with 1's for the pixels of the areas that are too small
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < threshold:
            # print(area)
            componentMask = (labels == i).astype(np.uint8)
            mask = cv2.bitwise_or(mask, componentMask)

    predictionChannels = tf.unstack(prediction, axis=-1)
    # we add the mask of the too small pixels to the mask of the background
    # values multiplied by 10 to make sure that those pixels are assigned to black
    updatedBackground = np.add(predictionChannels[4], mask * 100.0)

    # plt.imshow(updatedBackground, cmap='gray')
    # plt.show()

    updatedPrediction = tf.stack([predictionChannels[0], predictionChannels[1],
                                  predictionChannels[2], predictionChannels[3],
                                  updatedBackground], axis=-1)
    return updatedPrediction

def detect_alpha(prediction):
    # set each red, violet or green area that is to small to be a particle as black
    green_threshold = 95

    filtered_green = filterGreenToBlack(prediction, green_threshold)

    return filtered_green
