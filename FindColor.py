import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from CalcMetric import calc_metric


def detect_color(predict_color: np.ndarray) -> str:
    colors_to_detect = {
        'red': (127, 63, 63),
        'black': (34, 34, 34),
        'blue': (30, 30, 127),
        'blue_cyan': (30, 127, 127),
        'green': (40, 140, 40),
        'white_silver': (170, 170, 170),
        'yellow': (127, 127, 30)
    }
    min_error = 10e12
    for color in colors_to_detect:
        color_np = np.array(colors_to_detect[color])
        metric = (color_np - predict_color) ** 2
        metric = np.sum(metric) ** (1 / 2)
        if metric < min_error:
            min_error = metric
            target_color = color
    return target_color


def scale_image(img, scale_factor: int = 900):
    scaler = max(img.shape[0], img.shape[1]) / scale_factor
    if scaler == 0:
        return img
    return cv2.resize(img, (int(img.shape[1] / scaler),
                            int(img.shape[0] / scaler)), cv2.INTER_NEAREST)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def find_color(input_dir: str, output_cars: str = "output_color.csv", show_result: bool = True):
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
        # cv2.imshow('test', merged_image)
        # cv2.waitKey()
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

        # Inference
        inference_result = model(merged_image)
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)

        # Checking cars in result
        car = False
        max_square = 0
        for tensor in inference_result.xywh[0]:
            coor = tensor.numpy()
            type_of_det = inference_result.names[int(coor[-1])]
            if type_of_det in ['car', 'truck', 'bus']:
                coor = coor.astype('int32')
                w_test, h_test = coor[2], coor[3]
                # I do this in the way there isn't only 1 car in images
                if w_test * h_test > max_square:
                    max_square = w_test * h_test
                    x, y, w, h = coor[0], coor[1], coor[2], coor[3]
                    x, y = x - w // 2, y - h // 2
                car = True

        if not car:
            predict_color = calc_metric(merged_image, 0, 0, merged_image.shape[1], merged_image.shape[0],
                                        show_result=False)
        else:
            # Make image smaller to avoid colors of environment
            x, y, w, h = x + w // 3, y + h // 3, int(w * 0.45), int(h * 0.45)
            predict_color = calc_metric(merged_image, x, y, w, h, show_result=False)

        predict_color = np.array(predict_color, dtype='uint8')

        min = 10e8
        target_color = detect_color(predict_color)
        all_results.append([f'{str(img_index).zfill(5)}.jpg', target_color])

    # Saving results
    results = pd.DataFrame(all_results)
    results.to_csv(output_cars, index=False, header=False)

    # Check metrics
    validation = pd.read_csv(input_dir + 'colors.csv', header=None)
    for_metric = pd.merge(results, validation, on=0)
    accuracy = len(for_metric[for_metric['1_x'] == for_metric['1_y']]) / len(for_metric)
    failed_images = for_metric[for_metric['1_x'] != for_metric['1_y']]

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Failed images: {failed_images}")


if __name__ == '__main__':
    find_color('dataset/', show_result=True)
