import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Fitting the model is presented in the file : CV50ModelTrain.ipynb
# model = tf.keras.models.load_model('VezdekodCV50Model.h5')

# К сожалению у меня не хватило времени, чтобы обучить свою легковесную модель для это задачи
# Поэтому я сошёл с ума, взял голову от InceptionV3 и лишь дообучил развертку :D
# Моделька получилась жирной больше 100 мб, поэтому её не получается залить на git
model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False, weights='imagenet', input_shape=(256, 256, 3))
model.trainable = False

inputs = tf.keras.Input(shape=(256, 256, 3))
augment_layers = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
x = model(augment_layers, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(x)
x = tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(x)
x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)
new_model = tf.keras.Model(inputs=inputs, outputs=predictions)

new_model.load_weights('weightsInceptionV3.h5')

num_to_answer = {
    0: 'Off',
    1: 'On',
    2: 'no_cars'
}


def check_stop_signals(input_dir: str = 'stop_signals/', output_file: str = 'stop_signal_output.csv'):
    # Check dirs
    if not os.path.exists(input_dir):
        print(f'There is no directory with name: {input_dir}')

    description = pd.read_csv(input_dir + '/description.csv')
    all_image_indexes = description.full_image_index.unique()

    all_results = []
    for img_index in tqdm(all_image_indexes):
        r_path = input_dir + 'data/' + description[
            (description.full_image_index == img_index) & (description.color == 'r')].image_path.values[0]
        g_path = input_dir + 'data/' + description[
            (description.full_image_index == img_index) & (description.color == 'g')].image_path.values[0]
        b_path = input_dir + 'data/' + description[
            (description.full_image_index == img_index) & (description.color == 'b')].image_path.values[0]

        r_channel = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
        g_channel = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
        b_channel = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
        merged_image = cv2.merge([b_channel, g_channel, r_channel])

        merged_image = cv2.resize(merged_image, (256, 256))
        cv2.imwrite('current.jpg', merged_image)
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)
        merged_image = np.expand_dims(merged_image, axis=0)
        prediction = new_model.predict(merged_image)
        answer = num_to_answer[np.argmax(prediction)]
        all_results.append((f'{str(img_index).zfill(5)}.jpg', answer))

    results = pd.DataFrame(all_results)
    results.to_csv(output_file, index=False, header=False)

    # Check metrics
    validation = pd.read_csv(input_dir + 'train.csv', header=None)
    for_metric = pd.merge(results, validation, on=0)
    accuracy = len(for_metric[for_metric['1_x'] == for_metric['1_y']]) / len(for_metric)
    failed_images = for_metric[for_metric['1_x'] != for_metric['1_y']][0].values

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Failed images: {failed_images}")


if __name__ == '__main__':
    check_stop_signals()
