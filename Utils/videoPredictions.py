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
import tensorflow.keras.preprocessing.image as ti

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_name = "C:/Users/Nicola/Desktop/Uni/ccp/TrainedModels/attention_transfer_model_clean/"
unet = tf.keras.models.load_model(model_name, compile=False)

# compile manually
unet.compile(optimizer="adam", loss=loss.pce_dice_loss, metrics=[loss.dice_coef])

path = "C:/Users/Nicola/Desktop/video_copies/2h24m19s_P_5_images/"
test_images_paths = sorted(glob(os.path.join(path, "images/*")))

output_path = "C:/Users/Nicola/Desktop/video_copies/proton_video_images2/"

plt.figure()

start = time.process_time()
i = 196
for image_path in test_images_paths[196:]:
    image = image_utils.parse_image(image_path)

    prediction = unet.predict(tf.stack([image]))
    prediction = tf.reshape(prediction, [992, 1312, 5])

    # plt.imshow(utils_and_loss.create_mask(prediction))
    # plt.show()

    v_filtered = detect_V.detect_V(prediction)
    # proton_filtered = detect_protons.detect_protons(v_filtered)
    # post_processed = post_processing.detect_alpha(proton_filtered)
    post_processed = post_processing.majority_vote(v_filtered)
    post_processed = post_processing.filterRedToBlack(post_processed, 231)
    post_processed = post_processing.filterGreenToBlack(post_processed, 95)

    mask = image_utils.create_mask(post_processed)
    output = tf.math.add(mask * 0.8, image)
    plt.axis('off')
    plt.imshow(output)
    # plt.savefig(output_path + str(i), bbox_inches="tight")
    # ti.save_img(output_path + str(i) + ".jpg", output)
    plt.savefig(output_path + str(i), bbox_inches="tight", pad_inches=0.0)
    i += 1
    # plt.show()

    print((time.process_time() - start) / 10.0)
