import math
import numpy as np
from skimage.morphology import skeletonize
import cv2
import tensorflow as tf
import os
import scipy.spatial
from sklearn.metrics import mean_absolute_error
from skimage.draw import line
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
import PostProcessing.post_processing as post_processing
import Utils.image_utils as image_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### V-DETECTION ###

# given a prediction this returns a prediction where every particle that has some violet in it is colored violet
def get_V_candidates(prediction):
    prediction_channels = tf.unstack(prediction, axis=-1)

    # stack the predictions of the big particles and background, so that electrons are not considered
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

    V_candidates_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)

    # for every connected component we create a mask, multiply it with the arg max labels of the big particles
    # and check if the connected component has violet pixels in it
    for i in range(1, numLabels):
        # returns a mask that has 1's only where the current connected component is
        componentMask = (labels == i).astype("uint8")

        component_labels = componentMask * argmax_mask

        violet_count = np.count_nonzero(component_labels == 3)

        # if the current particle has violet in it then the V probabilities of all the pixels in it
        # are set to a very high value
        if violet_count > 0:
            V_candidates_mask = np.add(V_candidates_mask, componentMask * 10)

    V_candidates_mask = tf.cast(V_candidates_mask, tf.float32)

    # build the updated prediction
    V_candidates_prediction = tf.stack([prediction_channels[0], prediction_channels[1],
                                         prediction_channels[2], V_candidates_mask,
                                         prediction_channels[4]], axis=-1)

    return V_candidates_prediction


# connects the endpoints of skeleton lines that are close to each other
def connect_violet(labels, numLabels, skeleton):
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

                # if the distance is small enough, then the lines are connected
                if min_distance <= 143:
                    # gets the coordinates of the close endpoints
                    where = np.where(distances == min_distance)

                    a = i_pixels[where[0]][0]
                    b = k_pixels[where[1]][0]

                    # draws a line between the endpoints
                    rr, cc = line(a[0], a[1], b[0], b[1])
                    i_line = (labels == i+1).astype(np.uint8)
                    k_line = (labels == k+1).astype(np.uint8)
                    extension = np.add(i_line, k_line)
                    extension[rr, cc] = 1

                    skeleton[rr, cc] = 1

    return skeleton


# fits a V shape to the skeleton line and checks if it fulfills the requirements
def fit_V(labels, numLabels):
    # initialize image that will contain the skeletons of the actual V particles
    v_lines = np.zeros(shape=[992, 1312], dtype=np.uint8)

    # fit a V for every skeleton line
    for j in range(1, numLabels):
        # get the coordinates of the pixels of the current skeleton line
        line_pixels = np.argwhere(labels == j)
        line_pixels = [tuple(x) for x in line_pixels]


        # get the min/max y/x values
        min_y = min(line_pixels, key=lambda t: t[0])[0]
        max_y = max(line_pixels, key=lambda t: t[0])[0]
        min_x = min(line_pixels, key=lambda t: t[1])[1]
        max_x = max(line_pixels, key=lambda t: t[1])[1]

        # calculate the difference between the max and min y/x values in order to
        # get the dimensions that are needed to build an image in which the current skeleton line
        # can be rotated
        y_diff = abs(max_y - min_y)
        x_diff = abs(max_x - min_x)

        if x_diff > y_diff:
            max_diff = x_diff + 1
        else:
            max_diff = y_diff + 1

        # dimensions are max_diff * 3 so that the current skeleton line is in the middle of the image
        # and rotation is always possible without part of the line being outside of the dimensions
        v_image = np.zeros(shape=[max_diff*3, max_diff*3], dtype=np.uint8)

        # draw the current skeleton line in the image that will be used for rotation by
        # shifting the pixels and drawing them into the middle of the image
        for pixel in line_pixels:
            pixel = (pixel[0] - min_y, pixel[1] - min_x)
            pixel = (pixel[0] + max_diff, pixel[1] + max_diff)
            v_image[pixel[0]][pixel[1]] = 1

        # used for computing the rotation matrix
        height = max_diff*3
        width = max_diff*3
        center = (width/2, height/2)

        is_V = False
        # rotate the current line 30 times by 12 degrees to make sure that the best fit is found
        for r in range(30):
            # compute rotation
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=r * 12, scale=1)
            rotated_v_image = cv2.warpAffine(src=v_image, M=rotate_matrix, dsize=(width, height))

            # compute the dimensions of the rotated line and draw into an image that has exactly the dimensions
            # of the line, this makes sure that the fitted V is as large as the rotated line
            current_line_pixels = np.argwhere(rotated_v_image == 1)
            current_line_pixels = [tuple(x) for x in current_line_pixels]

            current_min_y = min(current_line_pixels, key=lambda t: t[0])[0]
            current_max_y = max(current_line_pixels, key=lambda t: t[0])[0]
            current_min_x = min(current_line_pixels, key=lambda t: t[1])[1]
            current_max_x = max(current_line_pixels, key=lambda t: t[1])[1]

            current_y_diff = abs(current_max_y - current_min_y)
            current_x_diff = abs(current_max_x - current_min_x)

            current_v_image = np.zeros(shape=[current_y_diff + 1, current_x_diff + 1], dtype=np.uint8)

            current_shifted_line_pixels = []
            for pixel in current_line_pixels:
                pixel = (pixel[0] - current_min_y, pixel[1] - current_min_x)
                current_shifted_line_pixels.append(pixel)

            for pixel in current_shifted_line_pixels:
                current_v_image[pixel[0]][pixel[1]] = 1

            # retrieve the coordinates of the pixels of the current skeleton line
            y_train_original, X_train_original = np.where(current_v_image == 1)

            # standardize the coordinates
            X_mean = np.mean(X_train_original)
            X_std_dev = np.std(X_train_original)
            X_standardized = np.array([(x - X_mean) / X_std_dev for x in X_train_original])

            y_mean = np.mean(y_train_original)
            y_std_dev = np.std(y_train_original)
            y_standardized = np.array([(y - y_mean) / y_std_dev for y in y_train_original])

            # calculate the knots for the spline regression fit
            max_X_standardized = np.max(X_standardized)
            min_X_standardized = np.min(X_standardized)
            middle_X_standardized = (max_X_standardized + min_X_standardized) / 2

            # rare case when we have a straight vertical line
            if max_X_standardized == min_X_standardized:
                continue

            # reshape the coordinates
            X_standardized = X_standardized.reshape(-1, 1)
            y_standardized = y_standardized.reshape(-1, 1)

            # make the spline regression fit, such that it computes a V shape
            knots = [[min_X_standardized], [middle_X_standardized], [max_X_standardized]]
            model = make_pipeline(SplineTransformer(knots=knots, degree=1), Ridge(alpha=1e-3))
            try:
                model.fit(X_standardized, y_standardized)
            except:
                print("fit error")
                continue
            # get the y-coordinates of the V shape
            pred_standardized = model.predict(X_standardized)

            # compute the mean absolute error between the skeleton line and the V shape
            mae = mean_absolute_error(y_standardized, pred_standardized)

            # invert the standardization to calculate the angle of the V shape
            X_unstandardized = np.array([(x[0] * X_std_dev) + X_mean for x in X_standardized])
            middle_X_unstandardized = (np.max(X_unstandardized) + np.min(X_unstandardized)) / 2
            y_unstandardized = np.array([(y[0] * y_std_dev) + y_mean for y in y_standardized])
            pred_unstandardized = np.array([(y[0] * y_std_dev) + y_mean for y in pred_standardized])

            # make a list with the coordinates of the V shape
            pred_and_x = []
            for i in range(len(pred_unstandardized)):
                pred_and_x.append((pred_unstandardized[i], X_unstandardized[i]))

            # calculate the left and right endpoints of the V shape and it's "tip"
            left = min(pred_and_x, key=lambda t: t[1])
            right = max(pred_and_x, key=lambda t: t[1])

            middle = (0, 0)
            for coordinates in pred_and_x:
                if coordinates[1] <= (middle_X_unstandardized):
                    if coordinates[1] >= middle[1]:
                        middle = coordinates

            # calculate distances between those points, allows us to calculate the angle of the V shape
            c = math.dist(middle, left)
            b = math.dist(middle, right)
            a = math.dist(left, right)

            try:
                angle = round(math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))))
            except:
                print("angle error")
                continue

            # if the characteristics of the fitted V shape fulfill those of a V shape that has been fitted to the
            # skeleton line of a real V particle then the current skeleton line corresponds to a real V particle
            # and we can stop
            if mae <= 0.5:
                if b > 59 and c > 59:
                    if 15 <= angle <= 139:
                        is_V = True
                        break

        # if it passed then add the current skeleton line to the image with the skeleton lines of the real V particles
        if is_V:
            currentLine = (labels == j).astype("uint8")
            v_lines = cv2.bitwise_or(currentLine, v_lines)

    # return the image that contains the skeleton lines of the actual V particles
    return v_lines


# given a prediction this returns a mask that contains the real V's
def get_V(prediction):
    # create a binary image of the prediction for the V particles and skeletonize it
    binary = image_utils.make_binary(prediction, color="violet")
    skeleton = skeletonize(binary.numpy()).astype(np.uint8)

    # extract individual lines
    connected_components = cv2.connectedComponentsWithStats(image=skeleton, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    # connect lines that are not on a straight line
    connect_lines = connect_violet(labels, numLabels, skeleton)

    # extract individual connected lines
    connected_components = cv2.connectedComponentsWithStats(image=connect_lines, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    # check which lines correspond to actual V particles by fitting a V shape to them
    v_lines = fit_V(labels, numLabels)


    # turn the mask of V lines into a mask of V's
    connected_components = cv2.connectedComponentsWithStats(image=binary.numpy(), connectivity=8, ltype=cv2.CV_32S)
    binary_numLabels = connected_components[0]
    binary_labels = connected_components[1]

    real_v = v_lines * binary_labels
    real_v_labels = []
    for i in range(1, binary_numLabels):
        label_count = np.count_nonzero(real_v == i)
        if label_count > 0:
            real_v_labels.append(i)

    v_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    for i in real_v_labels:
        curr_v = (binary_labels == i).astype(np.uint8)
        v_mask = cv2.bitwise_or(curr_v, v_mask)

    return v_mask


# given a prediction and a mask with the real V's this returns a prediction with only real V's
def filter_V(prediction, mask):
    # get a binary image of all the particles minus the electrons
    binary = image_utils.make_binary(prediction, color="green_red_violet")

    # extract all the single particles
    connected_components = cv2.connectedComponentsWithStats(image=binary, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    # get the labels of the particles, for which the skeleton line passed the V detection
    real_v = mask * labels
    real_v_labels = []
    for i in range(1, numLabels):
        label_count = np.count_nonzero(real_v == i)
        if label_count > 0:
            real_v_labels.append(i)

    # build an image that only contains the real V particles
    v_mask = np.zeros(shape=[992, 1312], dtype=np.uint8)
    for i in real_v_labels:
        curr_v = (labels == i).astype(np.uint8)
        v_mask = cv2.bitwise_or(curr_v, v_mask)

    not_v_mask = (v_mask == 0).astype(np.uint8) * binary
    invert_not_v_mask = (not_v_mask == 0).astype(np.uint8)

    channels = tf.unstack(prediction, axis=-1)
    # set the probabilities of the pixels of the real V particles to a very high value
    filtered_v = v_mask * 100
    # set probabilities for electrons and background to 0, since we know that we only checked pixels that belong to
    # big particles
    adjusted_electrons = channels[1] * invert_not_v_mask
    adjusted_background = channels[4] * invert_not_v_mask

    # return the adjusted prediction
    v_filtered_prediction = tf.stack([channels[0], adjusted_electrons,
                                           channels[2], filtered_v,
                                           adjusted_background], axis=-1)
    return v_filtered_prediction

# annotate each particle with some violet pixels as a V particle candidate and filter out all candidates
# that are too small to be particles
def pre_processing_V(prediction):
    v_candidates = get_V_candidates(prediction)

    violet_threshold = 1337
    filtered_violet = post_processing.filterViolet(v_candidates, violet_threshold)

    return filtered_violet

def detect_V(prediction):
    pre_processed = pre_processing_V(prediction)

    v_lines = get_V(pre_processed)

    v_filtered = filter_V(prediction, v_lines)

    return v_filtered
