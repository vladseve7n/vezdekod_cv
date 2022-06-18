import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def scale_image(img, scale_factor: int = 900):
    scaler = max(img.shape[0], img.shape[1]) / scale_factor
    if scaler == 0:
        return img
    return cv2.resize(img, (int(img.shape[1] / scaler),
                            int(img.shape[0] / scaler)), cv2.INTER_NEAREST)


def merge_channels(input_dir: str = 'dataset/', output_dir: str = 'merged_img/',
                   show_result: bool = True) -> None:
    """
    function for merging r,g,b channels
    :param input_dir: Directory for dataset. Example: 'dataset/'.
    :param output_dir: Directory to save merged images. Example: 'merged_img/'
    :param show_result: Flag for showing result
    :return: None
    """
    # Check dirs
    if not os.path.exists(input_dir):
        print(f'There is no directory with name: {input_dir}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    description = pd.read_csv(input_dir + '/description.csv')
    all_image_indexes = description.full_image_index.unique()

    cnt = 0
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

        if show_result:
            result_img = np.hstack((cv2.cvtColor(r_channel, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(g_channel, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(b_channel, cv2.COLOR_GRAY2BGR),
                                    merged_image))
            result_img = scale_image(result_img, 1200)
            cv2.imshow('result', result_img)
            cv2.waitKey()

        cv2.imwrite(output_dir + f'{str(img_index).zfill(5)}.jpg', merged_image)
        cnt += 1

    print(f'Number of merged images - {cnt}')


if __name__ == '__main__':
    merge_channels('dataset/', 'result/', show_result=False)
