import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import os
import time
import loss
import image_utils
import PostProcessing.post_processing as post_processing
import PostProcessing.detect_protons as detect_protons
import PostProcessing.detect_V as detect_V
# import PostProcessingLSC.post_processing_LSC as post_processing
# import PostProcessingLSC.detect_protons_LSC as detect_protons
# import PostProcessingLSC.detect_V_LSC as detect_V
import numpy as np
import cv2

# turns a prediction into a mask
def create_single_color_mask(binary_mask, color):
    # input is a tensor that contains the class probability maps

    # colors for the predicted mask, need to be divided by 255.0 since we work with float colors
    colors = [
        [0.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],  # Green
        [0.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],  # Blue
        # [227.0 / 255.0, 26.0 / 255.0, 28.0 / 255.0],  # Red
        [255.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],  # Yellow
        [0.0, 0.0, 0.0]  # Black
    ]

    binary_mask = tf.cast(binary_mask, tf.float32)
    mask = tf.stack([binary_mask, binary_mask, binary_mask], axis=-1) * colors[color]

    return mask

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

def parse_mask(name):
    mask = np.load(name)['arr_0']
    mask = tf.cast(mask, tf.float32)
    return mask


# returns a binary mask with the big particles
def get_particle_mask(prediction):
    # input is a tensor that contains the class probability maps

    # green, blue, red, violet, black
    class_indices = [0, 1, 2, 3, 4]
    # returns a 2d class map where in every pixel stands the class index of the
    # class that had the highest probability for that pixel
    argmax_mask = tf.argmax(prediction, axis=-1)

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
    green = channels[0].numpy() * 0
    blue = channels[1].numpy() * 1
    red = channels[2].numpy() * 0
    violet = channels[3].numpy() * 0
    black = channels[4].numpy() * 0

    # add all the RGB class maps together to get the final mask
    mask = np.add(red, green)
    mask = np.add(mask, violet)
    mask = np.add(mask, blue)
    mask = np.add(mask, black)

    return mask

# calculates how much area of the predicted mask aligns with the annotated mask
def calculate_overlap(annotated_mask, predicted_mask):
    overlap_mask = cv2.bitwise_and(annotated_mask, predicted_mask)

    annotated_mask_area = np.count_nonzero(annotated_mask)
    overlap_area = np.count_nonzero(overlap_mask)

    return round(overlap_area / annotated_mask_area, 2)

# calculates how much area of the predicted mask aligns with the annotated mask
def calculate_dice_coefficient(annotated_mask, predicted_mask):
    overlap_mask = cv2.bitwise_and(annotated_mask, predicted_mask)

    annotated_mask_area = np.count_nonzero(annotated_mask)
    predicted_mask_area = np.count_nonzero(predicted_mask)
    overlap_area = np.count_nonzero(overlap_mask)

    dice = (2 * overlap_area) / (annotated_mask_area + predicted_mask_area)

    return round(dice, 2)


# builds a separate image for every particle in the frame
def get_particles(mask):
    # get the individual particles
    connected_components = cv2.connectedComponentsWithStats(image=mask, connectivity=8, ltype=cv2.CV_32S)
    numLabels = connected_components[0]
    labels = connected_components[1]

    # build an image for each
    particles = []
    for i in range(1, numLabels):
        particle_image = (labels == i).astype("uint8")
        particles.append(particle_image)

    return particles



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
model_name = "C:/Users/Nicola/Desktop/Uni/ccp/TrainedModels/attention_transfer_model_clean/"
unet = tf.keras.models.load_model(model_name, compile=False)

# compile manually
unet.compile(optimizer="adam", loss=loss.pce_dice_loss, metrics=[loss.dice_coef])

path = "C:/Users/Nicola/Desktop/Uni/ccp/Data/full_data_set_augmented_validation/"
pairs = make_pairs(path, 'val')


plt.figure()


start = time.process_time()
i = 1
errors = []
for pair in pairs:
    image = image_utils.parse_image(pair[0])

    mask = parse_mask(pair[1])

    prediction = unet.predict(tf.stack([image]))
    prediction = tf.reshape(prediction, [992, 1312, 5])
    majority_vote_prediction = post_processing.majority_vote(prediction)

    # plt.imshow(image_utils.create_mask(mask))
    # plt.show()

    # v_filtered = detect_V.detect_V(prediction)
    # proton_filtered = detect_protons.detect_protons(v_filtered)
    # prediction_post_processed = post_processing.detect_alpha(proton_filtered)

    # v_filtered = detect_V.detect_V(prediction)
    # prediction_post_processed = post_processing.majority_vote(v_filtered)

    # plt.imshow(image_utils.create_mask(majority_vote_prediction))
    # plt.show()

    annotated_binary = get_particle_mask(mask)
    # plt.imshow(annotated_binary, cmap='gray')
    # plt.show()

    predicted_binary = get_particle_mask(prediction)
    # plt.imshow(predicted_binary, cmap='gray')
    # plt.show()

    # overlap = calculate_overlap(annotated_binary, predicted_binary)
    # errors.append(overlap)

    dice_coefficient = calculate_dice_coefficient(annotated_binary, predicted_binary)
    errors.append(dice_coefficient)


    plt.title(dice_coefficient)
    # plt.imshow(tf.math.add(image_utils.create_mask(prediction_post_processed)*0.5, image_utils.create_mask(mask)*0.5))
    overlap_binary = cv2.bitwise_and(annotated_binary, predicted_binary)
    overlap_mask = create_single_color_mask(overlap_binary, 0)
    # plt.imshow(overlap_mask)
    # plt.show()
    no_overlap_annotated_binary = cv2.bitwise_xor(annotated_binary, overlap_binary)
    no_overlap_annotated_mask = create_single_color_mask(no_overlap_annotated_binary, 1)
    # plt.imshow(no_overlap_annotated_mask)
    # plt.show()
    no_overlap_predicted_binary = cv2.bitwise_xor(predicted_binary, overlap_binary)
    no_overlap_predicted_mask = create_single_color_mask(no_overlap_predicted_binary, 2)
    # plt.imshow(no_overlap_predicted_mask)
    # plt.show()
    total_mask = tf.math.add(overlap_mask, no_overlap_annotated_mask * 0.75)
    total_mask = tf.math.add(total_mask, no_overlap_predicted_mask * 0.75)
    # plt.imshow(total_mask)
    # plt.show()
    
    # plt.imshow(tf.math.add(predicted_binary * 0.5, annotated_binary * 0.5), cmap='gray')
    # plt.imshow(tf.math.add(create_single_color_mask(prediction_post_processed, 1) * 0.5, create_single_color_mask(mask, 0) * 0.5))
    # plt.show()
 

    i += 1

    print((time.process_time() - start) / 10.0)
"""

# np.savez_compressed("C:/Users/Nicola/Desktop/object_detection_errors_electron.npz", errors)

object_detection_errors = np.load("C:/Users/Nicola/Desktop/object_detection_errors_big.npz")['arr_0']

print(np.mean(object_detection_errors))
print(np.std(object_detection_errors))
