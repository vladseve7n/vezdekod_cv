import os

import cv2
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

def scale_image(img, scale_factor: int = 900):
    scaler = max(img.shape[0], img.shape[1]) / scale_factor
    if scaler == 0:
        return img
    return cv2.resize(img, (int(img.shape[1] / scaler),
                            int(img.shape[0] / scaler)), cv2.INTER_NEAREST)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# car_cascade = cv2.CascadeClassifier('cars.xml')


def find_car(input_dir: str, output_cars: str = "output.csv", show_result: bool = True,
             threshold: float = 0.45):
    # Check dirs
    if not os.path.exists(input_dir):
        print(f'There is no directory with name: {input_dir}')

    description = pd.read_csv(input_dir + '/description.csv')
    all_image_indexes = description.full_image_index.unique()

    cnt = 0
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
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

        # Inference
        inference_result = model(merged_image)

        # Checking cars in result
        car = False
        for tensor in inference_result.xyxy[0]:
            coor = tensor.numpy()
            type_of_det = inference_result.names[int(coor[-1])]
            confidence = coor[-2]
            if type_of_det in ['car', 'truck', 'bus'] and confidence > threshold:
                car = True
                break

        all_results.append([f'{str(img_index).zfill(5)}.jpg', car])

        if show_result:
            yolov5_res = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
            for tensor in inference_result.xyxy[0]:
                coor = tensor.numpy()
                type_of_det = inference_result.names[int(coor[-1])]
                confidence = coor[-2]
                if type_of_det in ['car', 'truck', 'bus'] and confidence > threshold:
                    coor = np.int32(coor)
                    x1, y1, x2, y2 = coor[0], coor[1], coor[2], coor[3]
                    cv2.rectangle(yolov5_res, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

            yolov5_res = scale_image(yolov5_res, 500)
            cv2.imshow('YOLO result', yolov5_res)
            cv2.waitKey()
    # Saving results
    results = pd.DataFrame(all_results)
    results.to_csv(output_cars, index=False, header=False)

    # Check metrics
    validation = pd.read_csv(input_dir + 'val.csv', header=None)
    for_metric = pd.merge(results, validation, on=0)
    accuracy = len(for_metric[for_metric['1_x'] == for_metric['1_y']]) / len(for_metric)
    failed_images = for_metric[for_metric['1_x'] != for_metric['1_y']][0].values

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Failed images: {failed_images}")


if __name__ == '__main__':
    find_car('dataset/', show_result=False)
