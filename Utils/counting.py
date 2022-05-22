import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import os
import PostProcessing.post_processing as post_processing
import PostProcessing.detect_protons as detect_protons
import PostProcessing.detect_V as detect_V
import image_utils
import numpy as np
import cv2
from skimage.morphology import skeletonize, dilation, disk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.spatial
from skimage.draw import line

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# calculates how much area of a particle in the new frame overlaps with any particle in the old frame
def calculate_overlap(old_mask, new_particle):
    overlap_mask = cv2.bitwise_and(old_mask, new_particle)

    new_particle_area = np.count_nonzero(new_particle)
    overlap_area = np.count_nonzero(overlap_mask)

    return round(overlap_area / new_particle_area, 2)

# builds a separate image for every particle in the new frame
def get_new_particles(new_mask):
    # get the individual particles
    connected_components = cv2.connectedComponentsWithStats(image=new_mask, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    # build an image for each
    new_particles = []
    for i in range(1, numLabels):
        particle_image = (labels == i).astype("uint8")
        new_particles.append(particle_image)

    return new_particles

# counts how many new alpha particles have appeared between the previous frame and the current frame
def count_green(old_prediction, new_prediction):
    # build the binary masks for the alpha particles in the previous and current frame
    old_green_mask = image_utils.make_binary(old_prediction, 'green').numpy()
    new_green_mask = image_utils.make_binary(new_prediction, 'green').numpy()

    # retrieve the individual particles
    new_green_mask_particles = get_new_particles(new_green_mask)

    # count for how many particles in the current frame no overlap with particles in the previous frame occurs
    current_green_count = 0
    for particle in new_green_mask_particles:
        overlap = calculate_overlap(old_green_mask, particle)

        # if the percentage of overlapping area is 0 then we know that this is a new particle
        if overlap == 0.0:
            current_green_count += 1

    return current_green_count

# counts how many new proton particles have appeared between the previous frame and the current frame
def count_red(old_prediction, new_prediction):
    # build the binary masks for the proton particles in the previous and current frame
    old_red_mask = image_utils.make_binary(old_prediction, 'red').numpy()
    new_red_mask = image_utils.make_binary(new_prediction, 'red').numpy()

    # fill possible gaps between protons to prevent them from being counted as multiple particles
    old_red_mask = fill_proton_gaps(old_red_mask)
    new_red_mask = fill_proton_gaps(new_red_mask)

    # retrieve the individual particles
    new_green_mask_particles = get_new_particles(new_red_mask)

    # count for how many particles in the current frame no overlap with particles in the previous frame occurs
    current_green_count = 0
    for particle in new_green_mask_particles:
        overlap = calculate_overlap(old_red_mask, particle)

        # if the percentage of overlapping area is 0 then we know that this is a new particle
        if overlap == 0.0:
            current_green_count += 1

    return current_green_count

# counts how many new V particles have appeared between the previous frame and the current frame
def count_violet(old_prediction, new_prediction):
    # build the binary masks for the V particles in the previous and current frame
    old_violet_mask = image_utils.make_binary(old_prediction, 'violet').numpy()
    new_violet_mask = image_utils.make_binary(new_prediction, 'violet').numpy()

    # fill possible gaps between V particles to prevent them from being counted as multiple particles
    old_violet_mask = fill_V_gaps(old_violet_mask)
    new_violet_mask = fill_V_gaps(new_violet_mask)

    # retrieve the individual particles
    new_green_mask_particles = get_new_particles(new_violet_mask)

    # count for how many particles in the current frame no overlap with particles in the previous frame occurs
    current_green_count = 0
    for particle in new_green_mask_particles:
        overlap = calculate_overlap(old_violet_mask, particle)

        # if the percentage of overlapping area is 0 then we know that this is a new particle
        if overlap == 0.0:
            current_green_count += 1

    return current_green_count

# fills the gaps in a proton particle
def fill_proton_gaps(proton_mask):
    # get skeleton lines of the proton mask
    skeleton = skeletonize(proton_mask).astype(np.uint8)

    # image in which the line that connects the parts of the proton will be stored
    extension_image = np.zeros(shape=[992, 1312], dtype=np.uint8)

    # extract individual lines
    connected_components = cv2.connectedComponentsWithStats(image=skeleton, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

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
                if min_distance <= 204:
                    # gets the coordinates of the close endpoints
                    where = np.where(distances == min_distance)

                    a = i_pixels[where[0]][0]
                    b = k_pixels[where[1]][0]

                    # draws the connection line
                    rr, cc = line(a[0], a[1], b[0], b[1])
                    i_line = (labels == i + 1).astype(np.uint8)
                    k_line = (labels == k + 1).astype(np.uint8)
                    extension = np.add(i_line, k_line)
                    extension[rr, cc] = 1

                    coordinates_x, coordinates_y = np.where(extension == 1)
                    data = [list(x) for x in zip(coordinates_x, coordinates_y)]

                    # standardizes the data anc computes the linear regression fit
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data)

                    scaled_x = np.array([i[0] for i in scaled_data]).reshape(-1, 1)
                    scaled_y = np.array([i[1] for i in scaled_data]).reshape(-1, 1)

                    linReg = LinearRegression().fit(X=scaled_x, y=scaled_y)
                    pred = linReg.predict(scaled_x)

                    # if the connection results in a straight line, then the connecting line is saved
                    if mean_absolute_error(scaled_y, pred) < 0.36:
                        extension_image[rr, cc] = 1

    # dilate the connecting line such that overlap is ensured when we compare two protons with filled gaps
    footprint = disk(10)
    dilated_extension_image = dilation(extension_image, footprint)

    # add the dilated connecting line to the proton mask
    filled_proton_gaps = cv2.bitwise_or(proton_mask, dilated_extension_image)

    return filled_proton_gaps

# fills the gaps in a V particle
def fill_V_gaps(V_mask):
    # get skeleton lines of the V mask
    skeleton = skeletonize(V_mask).astype(np.uint8)

    # image in which the line that connects the parts of the V will be stored
    extension_image = np.zeros(shape=[992, 1312], dtype=np.uint8)

    # extract individual lines
    connected_components = cv2.connectedComponentsWithStats(image=skeleton, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

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
                if min_distance <= 150:
                    # gets the coordinates of the close endpoints
                    where = np.where(distances == min_distance)

                    a = i_pixels[where[0]][0]
                    b = k_pixels[where[1]][0]

                    # draws the connection line
                    rr, cc = line(a[0], a[1], b[0], b[1])
                    i_line = (labels == i + 1).astype(np.uint8)
                    k_line = (labels == k + 1).astype(np.uint8)
                    extension = np.add(i_line, k_line)
                    extension[rr, cc] = 1

                    # the connecting line is saved
                    extension_image[rr, cc] = 1

    # dilate the connecting line such that overlap is ensured when we compare two protons with filled gaps
    footprint = disk(10)
    dilated_extension_image = dilation(extension_image, footprint)

    # add the dilated connecting line to the proton mask
    filled_V_gaps = cv2.bitwise_or(V_mask, dilated_extension_image)

    return filled_V_gaps


# load the model for making the predictions
model_name = "C:/Users/Nicola/Desktop/Uni/ccp/TrainedModels/attention_transfer_model_clean/"
unet = tf.keras.models.load_model(model_name, compile=False)

# path to the images
path = "C:/Users/Nicola/Desktop/"
test_images_paths = sorted(glob(os.path.join(path, "NewImages/4m2s10FPS/images/*")))


# the initial prediction needs to be that everything is background, so that all the particles
# in the first actual frame are counted
zeros = tf.zeros([992, 1312], dtype=tf.float32)
ones = tf.ones([992, 1312], dtype=tf.float32)
old_prediction = tf.stack([zeros, zeros, zeros, zeros, ones], axis=-1)

# initialize the counts
total_green_count = 0
total_red_count = 0
total_violet_count = 0
i = 0
plt.figure()
for image_path in test_images_paths:
    # parse current image
    image = image_utils.parse_image(image_path)

    # make the prediction
    prediction = unet.predict(tf.stack([image]))
    prediction = tf.reshape(prediction, [992, 1312, 5])

    # use post processing to improve the prediction
    v_filtered = detect_V.detect_V(prediction)
    proton_filtered = detect_protons.detect_protons(v_filtered)
    new_prediction = post_processing.detect_alpha(proton_filtered)

    # count the particles
    current_green_count = count_green(old_prediction, new_prediction)
    current_red_count = count_red(old_prediction, new_prediction)
    current_violet_count = count_violet(old_prediction, new_prediction)

    # increase the counts
    total_green_count += current_green_count
    total_red_count += current_red_count
    total_violet_count += current_violet_count

    # plots the current prediction with the counts so far
    plt.title("Alpha: " + str(total_green_count) + " Proton: " + str(total_red_count) + " V: " + str(total_violet_count))
    plt.axis('off')
    plt.imshow(tf.math.add(image_utils.create_mask(new_prediction), image))
    # plt.savefig("C:/Users/Nicola/Desktop/count_results2/" + str(i), bbox_inches="tight")
    plt.show()
    i += 1

    old_prediction = new_prediction

print("Alpha count: " + str(total_green_count))
print("Proton count: " + str(total_red_count))
print("V particle count: " + str(total_violet_count))
