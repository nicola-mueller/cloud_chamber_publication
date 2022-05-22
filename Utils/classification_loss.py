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

def get_class_mask(prediction):
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
    green = channels[0].numpy() * 1
    blue = channels[1].numpy() * 4
    red = channels[2].numpy() * 2
    violet = channels[3].numpy() * 3
    black = channels[4].numpy() * 0

    # add all the RGB class maps together to get the final mask
    mask = np.add(red, green)
    mask = np.add(mask, violet)
    mask = np.add(mask, blue)
    mask = np.add(mask, black)

    return mask


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

def modify_prediction(prediction):
    binary_no_electrons = image_utils.make_binary(prediction, "green_red_violet")

    background = (binary_no_electrons == 0).astype('float32') * 1.0
    electron = binary_no_electrons.astype('float32') * 0.0
    alpha = binary_no_electrons.astype('float32') * 0.2
    proton = binary_no_electrons.astype('float32') * 0.3
    V = binary_no_electrons.astype('float32') * 0.5

    """
    plt.imshow(V, cmap='gray')
    plt.show()
    plt.imshow(proton, cmap='gray')
    plt.show()
    plt.imshow(alpha, cmap='gray')
    plt.show()
    plt.imshow(electron, cmap='gray')
    plt.show()
    plt.imshow(background, cmap='gray')
    plt.show()
    """

    modified_prediction = tf.stack([alpha, electron, proton, V, background], axis=-1)

    return modified_prediction



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
errors = [[], [], [], []]
for pair in pairs:
    image = image_utils.parse_image(pair[0])

    mask = parse_mask(pair[1])

    prediction = unet.predict(tf.stack([image]))
    prediction = tf.reshape(prediction, [992, 1312, 5])
    # majority_vote_prediction = post_processing.majority_vote(prediction)

    prediction = modify_prediction(prediction)

    # plt.imshow(image_utils.create_mask(prediction))
    # plt.show()

    v_filtered = detect_V.detect_V(prediction)
    proton_filtered = detect_protons.detect_protons(v_filtered)
    prediction_post_processed = post_processing.detect_alpha(proton_filtered)

    # v_filtered = detect_V.detect_V(prediction)
    # prediction_post_processed = post_processing.majority_vote(v_filtered)

    # plt.imshow(image_utils.create_mask(majority_vote_prediction))
    # plt.show()

    annotated = get_class_mask(mask)
    # plt.imshow(annotated, cmap='gray')
    # plt.show()
    annotated_binary = (annotated != 0).astype(np.uint8)
    # plt.imshow(annotated_binary, cmap='gray')
    # plt.show()

    predicted = get_class_mask(prediction_post_processed)
    # plt.imshow(predicted, cmap='gray')
    # plt.show()
    predicted_binary = (predicted != 0).astype(np.uint8)
    # plt.imshow(predicted_binary, cmap='gray')
    # plt.show()

    annotated_particles = get_particles(annotated_binary)

    for annotated_particle in annotated_particles:
        annotated_particle_pixel = np.argwhere(annotated_particle != 0)[0]
        annotated_particle_class = annotated[annotated_particle_pixel[0]][annotated_particle_pixel[1]]
        # print(annotated_particle_class)

        predicted_particle = cv2.bitwise_and(annotated_particle, predicted_binary)

        predicted_particle_pixels = np.argwhere(predicted_particle != 0)
        if len(predicted_particle_pixels) == 0:
            # errors[annotated_particle_class-1].append(-1)
            continue
        else:
            predicted_particle_pixel = predicted_particle_pixels[0]
        predicted_particle_class = predicted[predicted_particle_pixel[0]][predicted_particle_pixel[1]]
        # print(predicted_particle_class)

        # plt.imshow(annotated_particle, cmap='gray')
        # plt.title(str(annotated_particle_class) + " " + str(predicted_particle_class))
        # plt.show()

        errors[annotated_particle_class-1].append(predicted_particle_class)
    # break

    i += 1

    print((time.process_time() - start) / 10.0)

np.savez_compressed("C:/Users/Nicola/Desktop/empirical_data/alpha_errors_explicit.npz", errors[0])
np.savez_compressed("C:/Users/Nicola/Desktop/empirical_data/proton_errors_explicit.npz", errors[1])
np.savez_compressed("C:/Users/Nicola/Desktop/empirical_data/V_errors_explicit.npz", errors[2])
np.savez_compressed("C:/Users/Nicola/Desktop/empirical_data/electron_errors_explicit.npz", errors[3])
"""
alpha_errors = np.load("C:/Users/Nicola/Desktop/empirical_data/alpha_errors_explicit.npz")['arr_0']
proton_errors = np.load("C:/Users/Nicola/Desktop/empirical_data/proton_errors_explicit.npz")['arr_0']
V_errors = np.load("C:/Users/Nicola/Desktop/empirical_data/V_errors_explicit.npz")['arr_0']
electron_errors = np.load("C:/Users/Nicola/Desktop/empirical_data/electron_errors_explicit.npz")['arr_0']


alpha_misclassifications = np.count_nonzero((alpha_errors != 1))
number_of_alphas = len(alpha_errors)
print("% of alphas classified as other particle: " + str(alpha_misclassifications / number_of_alphas))

proton_misclassifications = np.count_nonzero((proton_errors != 2))
number_of_protons = len(proton_errors)
print("% of protons classified as other particle: " + str(proton_misclassifications / number_of_protons))

V_misclassifications = np.count_nonzero((V_errors != 3))
number_of_Vs = len(V_errors)
print("% of Vs classified as other particle: " + str(V_misclassifications / number_of_Vs))

print(electron_errors)
electron_misclassifications = np.count_nonzero((electron_errors != 4))
number_of_electrons = len(electron_errors)
print("% of electronss classified as other particle: " + str(electron_misclassifications / number_of_electrons))

print("\n")
alpha_proton_misclassifications = np.count_nonzero((alpha_errors == 2))
print("% of alpha misclassifications as protons: " + str(alpha_proton_misclassifications / alpha_misclassifications))
alpha_V_misclassifications = np.count_nonzero((alpha_errors == 3))
print("% of alpha misclassifications as Vs: " + str(alpha_V_misclassifications / alpha_misclassifications))
alpha_electron_misclassifications = np.count_nonzero((alpha_errors == 4))
print("% of alpha misclassifications as electronss: " + str(alpha_electron_misclassifications / alpha_misclassifications))

print("\n")
proton_alpha_misclassifications = np.count_nonzero((proton_errors == 1))
print("% of proton misclassifications as alphas: " + str(proton_alpha_misclassifications / proton_misclassifications))
proton_V_misclassifications = np.count_nonzero((proton_errors == 3))
print("% of proton misclassifications as Vs: " + str(proton_V_misclassifications / proton_misclassifications))
proton_electron_misclassifications = np.count_nonzero((proton_errors == 4))
print("% of proton misclassifications as electrons: " + str(proton_electron_misclassifications / proton_misclassifications))

print("\n")
V_alpha_misclassifications = np.count_nonzero((V_errors == 1))
print("% of V misclassifications as alphas: " + str(V_alpha_misclassifications / V_misclassifications))
V_proton_misclassifications = np.count_nonzero((V_errors == 2))
print("% of V misclassifications as protons: " + str(V_proton_misclassifications / V_misclassifications))
V_electron_misclassifications = np.count_nonzero((V_errors == 4))
print("% of V misclassifications as electrons: " + str(V_electron_misclassifications / V_misclassifications))

print("\n")
electron_alpha_misclassifications = np.count_nonzero((electron_errors == 1))
print("% of electron misclassifications as alphas: " + str(electron_alpha_misclassifications / electron_misclassifications))
electron_proton_misclassifications = np.count_nonzero((electron_errors == 2))
print("% of electron misclassifications as protons: " + str(electron_proton_misclassifications / electron_misclassifications))
electron_V_misclassifications = np.count_nonzero((proton_errors == 3))
print("% of electron misclassifications as Vs: " + str(electron_V_misclassifications / electron_misclassifications))


print(number_of_Vs)
print(number_of_alphas)
print(number_of_protons)
print(number_of_electrons)
print(number_of_Vs + number_of_electrons + number_of_alphas + number_of_protons)


number_of_particles = number_of_Vs + number_of_electrons + number_of_alphas + number_of_protons

alpha_misclassifications = np.count_nonzero((alpha_errors != 1))
number_of_alphas = len(alpha_errors)
print("% of alphas classified as other particle: " + str(alpha_misclassifications / number_of_particles))

proton_misclassifications = np.count_nonzero((proton_errors != 2))
number_of_protons = len(proton_errors)
print("% of protons classified as other particle: " + str(proton_misclassifications / number_of_particles))

V_misclassifications = np.count_nonzero((V_errors != 3))
number_of_Vs = len(V_errors)
print("% of Vs classified as other particle: " + str(V_misclassifications / number_of_particles))

electron_misclassifications = np.count_nonzero((electron_errors != 4))
number_of_electrons = len(electron_errors)
print("% of electronss classified as other particle: " + str(electron_misclassifications / number_of_particles))

print("\n")
alpha_proton_misclassifications = np.count_nonzero((alpha_errors == 2))
print("% of alpha misclassifications as protons: " + str(alpha_proton_misclassifications / number_of_particles))
alpha_V_misclassifications = np.count_nonzero((alpha_errors == 3))
print("% of alpha misclassifications as Vs: " + str(alpha_V_misclassifications / number_of_particles))
alpha_electron_misclassifications = np.count_nonzero((alpha_errors == 4))
print("% of alpha misclassifications as electronss: " + str(alpha_electron_misclassifications / number_of_particles))

print("\n")
proton_alpha_misclassifications = np.count_nonzero((proton_errors == 1))
print("% of proton misclassifications as alphas: " + str(proton_alpha_misclassifications / number_of_particles))
proton_V_misclassifications = np.count_nonzero((proton_errors == 3))
print("% of proton misclassifications as Vs: " + str(proton_V_misclassifications / number_of_particles))
proton_electron_misclassifications = np.count_nonzero((proton_errors == 4))
print("% of proton misclassifications as electrons: " + str(proton_electron_misclassifications / number_of_particles))

print("\n")
V_alpha_misclassifications = np.count_nonzero((V_errors == 1))
print("% of V misclassifications as alphas: " + str(V_alpha_misclassifications / number_of_particles))
V_proton_misclassifications = np.count_nonzero((V_errors == 2))
print("% of V misclassifications as protons: " + str(V_proton_misclassifications / number_of_particles))
V_electron_misclassifications = np.count_nonzero((V_errors == 4))
print("% of V misclassifications as electrons: " + str(V_electron_misclassifications / number_of_particles))

print("\n")
electron_alpha_misclassifications = np.count_nonzero((electron_errors == 1))
print("% of electron misclassifications as alphas: " + str(electron_alpha_misclassifications / number_of_particles))
electron_proton_misclassifications = np.count_nonzero((electron_errors == 2))
print("% of electron misclassifications as protons: " + str(electron_proton_misclassifications / number_of_particles))
electron_V_misclassifications = np.count_nonzero((proton_errors == 3))
print("% of electron misclassifications as Vs: " + str(electron_V_misclassifications / number_of_particles))
# """
