import cv2
import numpy as np
import pandas as pd


def scale_image(img, scale_factor: int = 900):
    scaler = max(img.shape[0], img.shape[1]) / scale_factor
    if scaler == 0:
        return img
    return cv2.resize(img, (int(img.shape[1] / scaler),
                            int(img.shape[0] / scaler)), cv2.INTER_NEAREST)


def calc_metric(image: np.ndarray, x: int, y: int, w: int, h: int,
                show_result: bool = True,
                type: str = 'cluster') -> tuple:
    part_of_image = image[y:y + h, x:x + w]
    source_part_of_image = part_of_image.copy()

    # First method
    # Using median value of image
    # Process image
    part_of_image = cv2.GaussianBlur(part_of_image, (3, 3), 0)
    part_of_image = cv2.blur(part_of_image, (256, 256))

    # Extract colors from processed image
    r_channel = int(np.median(part_of_image[:, :, 2]))
    g_channel = int(np.median(part_of_image[:, :, 1]))
    b_channel = int(np.median(part_of_image[:, :, 0]))

    answer = (b_channel, g_channel, r_channel)

    # Second method
    # Using clusterization
    # Process image
    data = np.reshape(source_part_of_image, (source_part_of_image.shape[0] * source_part_of_image.shape[1], 3))
    data = data.astype('float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    b, g, r = tuple(centers[0].astype('int'))
    cluster_answer = (int(b), int(g), int(r))

    if show_result:
        res = np.zeros_like(part_of_image)
        cv2.rectangle(res, (0, 0), (res.shape[1] // 2, res.shape[0]), answer, thickness=-1)
        cv2.rectangle(res, (res.shape[1] // 2, 0), (res.shape[1], res.shape[0]), cluster_answer, thickness=-1)
        cv2.putText(res, 'median', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 212, 124), thickness=2)
        cv2.putText(res, 'cluster', (res.shape[1] // 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 212, 124),
                    thickness=2)
        res = res.astype('uint8')
        res = np.hstack((source_part_of_image, np.zeros((res.shape[0], 20, 3), dtype='uint8'),
                         part_of_image, np.zeros((res.shape[0], 20, 3), dtype='uint8'), res))
        res = scale_image(res)
        cv2.imshow('result', res)
        cv2.waitKey()

    if type == 'median':
        return answer[::-1]
    else:
        return cluster_answer[::-1]


if __name__ == '__main__':

    input_dir = 'dataset/'
    description = pd.read_csv(input_dir + '/description.csv')
    all_image_indexes = description.full_image_index.unique()

    cnt = 0
    all_results = []
    for img_index in all_image_indexes:
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

        # Function using
        answer = calc_metric(merged_image, x=50, y=50, w=300, h=200, type='cluster')
        print(f'(R, G, B): {answer}')
