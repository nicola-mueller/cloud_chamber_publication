import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import os
import time
import loss
import image_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_name = "C:/Users/Nicola/Desktop/Uni/ccp/TrainedModels/attention_transfer_model_clean/"
unet = tf.keras.models.load_model(model_name, compile=False)

# compile manually
unet.compile(optimizer="adam", loss=loss.pce_dice_loss, metrics=[loss.dice_coef])

path = "C:/Users/Nicola/Desktop/Uni/ccp/Data/full_data_set_augmented_validation/"
test_images_paths = sorted(glob(os.path.join(path, "validation_images/*")))

output_path = "C:/Users/Nicola/Desktop/validation_results/"

plt.figure()
start = time.process_time()
i = 1
for image_path in test_images_paths:
    image = image_utils.parse_image(image_path)

    prediction = unet.predict(tf.stack([image]))
    prediction = tf.reshape(prediction, [992, 1312, 5])

    mask = image_utils.create_mask(prediction)
    output = tf.math.add(mask, image)
    plt.axis('off')
    plt.imshow(output)
    # plt.savefig(output_path + str(i), bbox_inches="tight")
    i += 1
    plt.show()

    print((time.process_time() - start) / 10.0)
