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
import numpy as np

def modify_prediction(prediction):
    binary_no_electrons = image_utils.make_binary(prediction, "green_red_violet")

    background = (binary_no_electrons == 0).astype('float32') * 1.0
    electron = binary_no_electrons.astype('float32') * 0.0
    alpha = binary_no_electrons.astype('float32') * 0.2
    proton = binary_no_electrons.astype('float32') * 0.3
    V = binary_no_electrons.astype('float32') * 0.5

    modified_prediction = tf.stack([alpha, electron, proton, V, background], axis=-1)

    return modified_prediction

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_name = "C:/Users/Nicola/Desktop/Uni/ccp/TrainedModels/attention_transfer_model_clean/"
unet = tf.keras.models.load_model(model_name, compile=False)

# compile manually
unet.compile(optimizer="adam", loss=loss.pce_dice_loss, metrics=[loss.dice_coef])

# path = "C:/Users/Nicola/Desktop/Uni/ccp/Data/full_data_set_augmented_validation/"
# test_images_paths = sorted(glob(os.path.join(path, "validation_images/*")))

path = "C:/Users/Nicola/Desktop/Uni/ccp/Data/full_data_set_augmented_validation/"
test_images_paths = sorted(glob(os.path.join(path, "validation_images/*")))

plt.figure()

start = time.process_time()
times = []
times_post_processing = []
i = 1
for image_path in test_images_paths:
    current_time = (time.process_time() - start) / 10.0

    image = image_utils.parse_image(image_path)

    prediction = unet.predict(tf.stack([image]))
    prediction = tf.reshape(prediction, [992, 1312, 5])

    output = tf.math.add(image_utils.create_mask(prediction)*0.8, image)

    prediction_time = ((time.process_time() - start) / 10.0) - current_time
    times.append(prediction_time)

    prediction = modify_prediction(prediction)

    v_filtered = detect_V.detect_V(prediction)
    proton_filtered = detect_protons.detect_protons(v_filtered)
    post_processed = post_processing.detect_alpha(proton_filtered)

    output_post_processed = tf.math.add(image_utils.create_mask(post_processed)*0.8, image)

    prediction_post_processed_time = ((time.process_time() - start) / 10.0) - current_time
    times_post_processing.append(prediction_post_processed_time)

    print(i)
    i += 1

print(np.mean(times))
print(np.mean(times_post_processing))
