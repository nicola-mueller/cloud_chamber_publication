import numpy as np
from skimage.morphology import skeletonize
import cv2
import tensorflow as tf
import os
import scipy.spatial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from skimage.draw import line
import PostProcessing.post_processing as post_processing
import Utils.image_utils as image_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### PROTON-DETECTION ###

# given a prediction this returns a prediction where every particle that has some red in it is colored red
def get_proton_candidates(prediction):
    prediction_channels = tf.unstack(prediction, axis=-1)

    # stack the predictions of the big particles and background
    big_particles = tf.stack([prediction_channels[4], prediction_channels[0],
                              prediction_channels[2], prediction_channels[3], prediction_channels[4]], axis=-1)

    # returns a mask that has in every pixel the index of the color that had the highest probability
    argmax_mask = tf.argmax(big_particles, axis=-1)

    # build an image out of the prediction for connected component analysis
    visualized_mask = post_processing.create_mask_no_electrons(prediction).numpy()

    # greyscale image required for connected components analysis
    mask_bw = cv2.cvtColor(visualized_mask, cv2.COLOR_BGR2GRAY)

    connected_components = cv2.connectedComponentsWithStats(image=mask_bw, connectivity=4, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_components

    proton_candidate_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    # for every connected component we create a mask, multiply it with the arg max labels of the big particles
    # and check if it has red in it
    for i in range(1, numLabels):
        # returns a mask that has 1's only where the current connected component is
        componentMask = (labels == i).astype("uint8")

        component_labels = componentMask * argmax_mask

        red_count = np.count_nonzero(component_labels == 2)

        # if the current particle has red in it then the proton probabilities of all the pixels in it
        # are set to a very high value
        if red_count > 0:
            proton_candidate_mask = np.add(proton_candidate_mask, componentMask * 10)

    proton_candidate_mask = tf.cast(proton_candidate_mask, tf.float32)

    # build the updated prediction
    proton_candidates_prediction = tf.stack([prediction_channels[0], prediction_channels[1],
                                             proton_candidate_mask, prediction_channels[3],
                                             prediction_channels[4]], axis=-1)

    return proton_candidates_prediction

# connects the endpoints of skeleton lines that lie on a straight line
def connect_red(labels, numLabels, skeleton):
    # retrieves the endpoints of every skeleton line
    endpoints = []
    for j in range(1, numLabels):
        line_pixels = np.argwhere(labels == j)
        # get the max distance between all the pixels of the current line
        line_pixel_distances = scipy.spatial.distance.cdist(line_pixels, line_pixels, 'cityblock')
        max_distance = line_pixel_distances.max()

        # get the pixels with max distance and use them as endpoints
        endpoints_coordinates = np.where(line_pixel_distances == max_distance)
        endpoint1 = line_pixels[endpoints_coordinates[0]][0]
        endpoint2 = line_pixels[endpoints_coordinates[1]][0]

        endpoints.append([endpoint1, endpoint2])

    # connect the lines that are close enough
    for i in range(0, len(endpoints)):
        for k in range(0, len(endpoints)):
            if k != i:
                # builds matrices that have 0's everywhere except where the endpoints are
                i_endpoints = endpoints[i]
                i_pixels = np.zeros(shape=[992, 1312], dtype=np.uint8)
                i_pixels[i_endpoints[0][0], i_endpoints[0][1]] = 1
                i_pixels[i_endpoints[1][0], i_endpoints[1][1]] = 1

                k_endpoints = endpoints[k]
                k_pixels = np.zeros(shape=[992, 1312], dtype=np.uint8)
                k_pixels[k_endpoints[0][0], k_endpoints[0][1]] = 1
                k_pixels[k_endpoints[1][0], k_endpoints[1][1]] = 1

                i_pixels = np.argwhere(i_pixels != 0)
                k_pixels = np.argwhere(k_pixels != 0)

                # calculates the distances between the endpoints
                distances = scipy.spatial.distance.cdist(i_pixels, k_pixels, 'cityblock')
                min_distance = distances.min()

                # if the distance is small enough, it fits a linear regression fit to the line and checks how large
                # the mean absolute error is, this is to ensure that we only connect proton lines in such a way
                # that it results in a straight line
                if min_distance <= 170:
                    # gets the coordinates of the close endpoints
                    where = np.where(distances == min_distance)

                    a = i_pixels[where[0]][0]
                    b = k_pixels[where[1]][0]

                    # draws the connection line between the endpoints
                    rr, cc = line(a[0], a[1], b[0], b[1])
                    i_line = (labels == i+1).astype(np.uint8)
                    k_line = (labels == k+1).astype(np.uint8)
                    extension = np.add(i_line, k_line)
                    extension[rr, cc] = 1

                    coordinates_x, coordinates_y = np.where(extension == 1)
                    data = [list(x) for x in zip(coordinates_x, coordinates_y)]

                    # standardizes the data and computes the linear regression fit
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data)

                    scaled_x = np.array([i[0] for i in scaled_data]).reshape(-1, 1)
                    scaled_y = np.array([i[1] for i in scaled_data]).reshape(-1, 1)

                    linReg = LinearRegression().fit(X=scaled_x, y=scaled_y)
                    pred = linReg.predict(scaled_x)

                    # if the connection results in a straight line, then it is used
                    if mean_absolute_error(scaled_y, pred) < 0.36:
                        skeleton[rr, cc] = 1

    return skeleton

# given a prediction this returns a mask that contains the real protons
def get_protons(prediction):
    # create a binary image of the prediction and skeletonize it
    binary = image_utils.make_binary(prediction, color="red")
    skeleton = skeletonize(binary.numpy()).astype(np.uint8)

    # extract individual lines
    connected_components = cv2.connectedComponentsWithStats(image=skeleton, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    # connect lines that are on a straight line
    connect_lines = connect_red(labels, numLabels, skeleton)

    # extract individual connected lines
    connected_components = cv2.connectedComponentsWithStats(image=connect_lines, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    # check for every connected line if it corresponds to a proton
    proton_lines = np.zeros(shape=[992, 1312], dtype=np.uint8)
    for i in range(1, numLabels):
        currentLine = (labels == i).astype("uint8")
        coordinates_x, coordinates_y = np.where(currentLine == 1)
        data = [list(x) for x in zip(coordinates_x, coordinates_y)]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        scaled_x = np.array([i[0] for i in scaled_data]).reshape(-1, 1)
        scaled_y = np.array([i[1] for i in scaled_data]).reshape(-1, 1)

        linReg = LinearRegression().fit(X=scaled_x, y=scaled_y)
        pred = linReg.predict(scaled_x)

        if len(scaled_x) > 103 and mean_absolute_error(scaled_y, pred) < 0.36:
            proton_lines = cv2.bitwise_or(currentLine, proton_lines)

    # turn the mask of proton lines into a mask of protons
    connected_components = cv2.connectedComponentsWithStats(image=binary.numpy(), connectivity=8, ltype=cv2.CV_32S)
    binary_numLabels = connected_components[0]
    binary_labels = connected_components[1]

    real_protons = proton_lines * binary_labels
    real_proton_labels = []
    for i in range(1, binary_numLabels):
        label_count = np.count_nonzero(real_protons == i)
        if label_count > 0:
            real_proton_labels.append(i)

    proton_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    for i in real_proton_labels:
        curr_proton = (binary_labels == i).astype(np.uint8)
        proton_mask = cv2.bitwise_or(curr_proton, proton_mask)

    return proton_mask

# given a prediction and a mask with the real protons this returns a prediction with only real protons
def filter_protons(prediction, mask):
    binary = image_utils.make_binary(prediction, color="green_red_violet")

    connected_components = cv2.connectedComponentsWithStats(image=binary, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    real_protons = mask * labels
    real_proton_labels = []
    for i in range(1, numLabels):
        label_count = np.count_nonzero(real_protons == i)
        if label_count > 0:
            real_proton_labels.append(i)

    proton_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    for i in real_proton_labels:
        curr_proton = (labels == i).astype(np.uint8)
        proton_mask = cv2.bitwise_or(curr_proton, proton_mask)

    not_proton_mask = (proton_mask == 0).astype(np.uint8) * binary
    invert_not_proton_mask = (not_proton_mask == 0).astype(np.uint8)

    channels = tf.unstack(prediction, axis=-1)
    filtered_protons = proton_mask * 100
    adjusted_electrons = channels[1] * invert_not_proton_mask
    adjusted_background = channels[4] * invert_not_proton_mask

    proton_filtered_prediction = tf.stack([channels[0], adjusted_electrons,
                                           filtered_protons, channels[3],
                                           adjusted_background], axis=-1)
    return proton_filtered_prediction

# filter the red areas that are too small to be protons
def pre_processing_P(prediction):
    proton_candidates = get_proton_candidates(prediction)

    red_threshold = 231
    filtered_red = post_processing.filterRed(proton_candidates, red_threshold)

    return filtered_red

def detect_protons(prediction):
    pre_processed = pre_processing_P(prediction)

    proton_lines = get_protons(pre_processed)

    proton_filtered = filter_protons(prediction, proton_lines)

    return proton_filtered
