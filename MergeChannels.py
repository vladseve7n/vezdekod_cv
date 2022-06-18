import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def scale_image(img, scale_factor: int = 900):
    scaler = max(img.shape[0], img.shape[1]) / scale_factor
    if scaler == 0:
        return img
    return cv2.resize(img, (int(img.shape[1] / scaler),
                            int(img.shape[0] / scaler)), cv2.INTER_NEAREST)


def merge_channels(input_dir: str = 'data/', output_dir: str = 'merged_img/',
                   show_result: bool = True) -> None:
    """
    function for merging r,g,b channels
    :param input_dir: Directory with image files. Example: 'data/'. Format of images: '00000_r.jpg'.
    :param output_dir: Directory to save merged images. Example: 'merged_img/'
    :param show_result: Flag for showing result
    :return: None
    """
    # Check dirs
    if not os.path.exists(input_dir):
        print(f'There is no directory with name: {input_dir}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dict_with_images = {}
    # Extracting image number and channels
    for path in glob(input_dir + '*.[jp][pn][g]'):
        # Convert tot Path for multiplatform
        img_path = Path(path)

        image_number = img_path.stem.split('_')[0]
        if image_number not in dict_with_images:
            dict_with_images[image_number] = {}

        image_channel = img_path.stem.split('_')[1]
        if image_channel not in dict_with_images[image_number]:
            dict_with_images[image_number][image_channel] = path

    # Merging images
    cnt = 0
    for img in tqdm(dict_with_images):

        # Check for all channels
        set_of_img_channels = set(list(dict_with_images[img].keys()))
        if {'r', 'g', 'b'} != set_of_img_channels:
            print(f"There aren't enough channels in image - {img}")
            continue

        r_channel = cv2.imread(dict_with_images[img]['r'], cv2.IMREAD_GRAYSCALE)
        g_channel = cv2.imread(dict_with_images[img]['g'], cv2.IMREAD_GRAYSCALE)
        b_channel = cv2.imread(dict_with_images[img]['b'], cv2.IMREAD_GRAYSCALE)
        merged_image = cv2.merge([b_channel, g_channel, r_channel])

        if show_result:
            result_img = np.hstack((cv2.cvtColor(r_channel, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(g_channel, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(b_channel, cv2.COLOR_GRAY2BGR),
                                    merged_image))
            result_img = scale_image(result_img, 1200)
            cv2.imshow('result', result_img)
            cv2.waitKey()

        cv2.imwrite(output_dir + f'{img}.jpg', merged_image)
        cnt += 1

    print(f'Number of merged images - {cnt}')


if __name__ == '__main__':
    merge_channels('data/', 'result/', show_result=True)
