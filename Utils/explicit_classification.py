import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
import os
import time
import loss
import image_utils
import PostProcessing.post_processing as post_processing
import PostProcessing.detect_protons as detect_protons
import PostProcessing.detect_V as detect_V

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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





model_name = "C:/Users/Nicola/Desktop/Uni/ccp/TrainedModels/attention_transfer_model_clean/"
unet = tf.keras.models.load_model(model_name, compile=False)

# compile manually
unet.compile(optimizer="adam", loss=loss.pce_dice_loss, metrics=[loss.dice_coef])

# path = "C:/Users/Nicola/Desktop/Uni/ccp/Data/full_data_set_augmented_validation/"
# test_images_paths = sorted(glob(os.path.join(path, "validation_images/*")))

path = "C:/Users/Nicola/Desktop/Uni/ccp/Data/full_data_set_augmented_validation/"
test_images_paths = sorted(glob(os.path.join(path, "validation_images/*")))

path = "C:/Users/Nicola/Desktop/"
test_images_paths = sorted(glob(os.path.join(path, "PP_Examples/*")))

output_path = "C:/Users/Nicola/Desktop/"

plt.figure()

start = time.process_time()
i = 1
for image_path in test_images_paths:
    image = image_utils.parse_image(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    prediction = unet.predict(tf.stack([image]))
    prediction = tf.reshape(prediction, [992, 1312, 5])
    modified_prediction = modify_prediction(prediction)

    plt.imshow(image_utils.create_mask(modified_prediction))
    plt.axis('off')
    plt.show()

    v_filtered = detect_V.detect_V(modified_prediction)
    plt.imshow(image_utils.create_mask(v_filtered))
    plt.axis('off')
    plt.show()
    proton_filtered = detect_protons.detect_protons(v_filtered)
    plt.imshow(image_utils.create_mask(proton_filtered))
    plt.axis('off')
    plt.show()
    post_processed = post_processing.detect_alpha(proton_filtered)
    plt.imshow(image_utils.create_mask(post_processed))
    plt.axis('off')
    plt.show()

    mask = image_utils.create_mask(post_processed)
    output = tf.math.add(mask*0.8, image)
    plt.axis('off')
    # plt.title(image_path.removeprefix("C:/Users/Nicola/Desktop/Uni/ccp/Data/full_data_set_augmented_validation/"))
    plt.imshow(output)
    # plt.savefig(output_path + str(i) + "_post", bbox_inches="tight")
    i += 1
    plt.show()

    print((time.process_time() - start) / 10.0)