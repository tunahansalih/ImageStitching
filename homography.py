import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt


def normalization_matrix(points):
    print('creating normalization matrix')
    nm = np.diag(np.ones(3))
    mean_of_points = np.mean(points, 0)
    dist_of_points = np.sum(np.sqrt(np.sum(np.square(points), 1))) / len(points)
    nm[0, 2] -= mean_of_points[0]
    nm[1, 2] -= mean_of_points[1]
    nm[:2, :] /= dist_of_points
    nm[:2, :] *= np.sqrt(2)
    return nm


def create_A_matrix_columns(im1point, im2point):
    print('creating A matrix column')
    col1 = np.array(
        [im1point[0], im1point[1], 1, 0, 0, 0, -im2point[0] * im1point[0], -im2point[0] * im1point[1], -im2point[0]])
    col2 = np.array(
        [0, 0, 0, im1point[0], im1point[1], 1, -im2point[1] * im1point[0], -im2point[1] * im1point[1], -im2point[1]])
    cols = np.stack((col1, col2))
    return cols


def create_A_matrix(im1Points, im2Points):
    print('Creating A matrix')
    A = []
    for i in range(option[0]):
        cols = create_A_matrix_columns(im1Points[i], im2Points[i])

        A.append(cols)

    return np.vstack(A)


def computeH(im1Points, im2Points, normalize=True):
    print('Computing homography matrix')
    if normalize:
        im1_normalization_matrix = normalization_matrix(im1Points)
        im2_normalization_matrix = normalization_matrix(im2Points)
        im1Points_normalized = np.matmul(im1_normalization_matrix, np.insert(im1Points, 2, 1, axis=1).T).T
        im2Points_normalized = np.matmul(im2_normalization_matrix, np.insert(im2Points, 2, 1, axis=1).T).T
        A = create_A_matrix(im1Points_normalized, im2Points_normalized)
        u, s, vh = np.linalg.svd(A)
        normalized_homography_matrix = np.reshape(vh[-1, :], (3, 3))
        homography_matrix = np.matmul(np.matmul(np.linalg.inv(im2_normalization_matrix), normalized_homography_matrix),
                                      im1_normalization_matrix) / normalized_homography_matrix[-1, -1]
    else:
        u, s, vh = np.linalg.svd(
            create_A_matrix(np.insert(im1Points, 2, 1, axis=1), np.insert(im2Points, 2, 1, axis=1)))
        homography_matrix = np.reshape(vh[-1, :], (3, 3))
    return homography_matrix / homography_matrix[2, 2]


def warp(image, H):
    print('Warping')

    y, x = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    coordinate_matrix = np.array([np.concatenate(np.stack((x, y), axis=2))], dtype=np.float)
    new_coordinates = np.reshape(cv2.perspectiveTransform(coordinate_matrix, H), (image.shape[0], image.shape[1], 2))

    new_img = np.zeros(BLENDED_IMAGE_SIZE)
    left_x_min_coordinate = int((BLENDED_IMAGE_SIZE[0] - image.shape[0]) / 2)
    left_y_min_coordinate = int((BLENDED_IMAGE_SIZE[1] - image.shape[1]) / 2)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_coordinate = new_coordinates[x, y] + [left_x_min_coordinate, left_y_min_coordinate]
            if 0 <= int(new_coordinate[0]) < BLENDED_IMAGE_SIZE[0] and 0 <= int(new_coordinate[1]) < BLENDED_IMAGE_SIZE[1]:
                new_img[int(new_coordinate[0]), int(new_coordinate[1])] = image[x, y]

    corner_coordinates = np.float32(
        [[0, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]], [image.shape[0], 0]]).reshape(-1, 1, 2)
    new_corner_coordinates = cv2.perspectiveTransform(corner_coordinates, H)
    return new_img.astype(np.uint8), new_corner_coordinates


def interpolate(new_image):
    print('Interpolating')
    mask = np.array(np.sum(new_image == 0, axis=2), dtype=np.uint8)
    interpolated_image = cv2.inpaint(src=new_image, inpaintMask=mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    return interpolated_image


def expand(img):
    new_img = np.zeros(BLENDED_IMAGE_SIZE)
    left_x_min_coordinate = int((BLENDED_IMAGE_SIZE[0] - img2.shape[0]) / 2)
    left_y_min_coordinate = int((BLENDED_IMAGE_SIZE[1] - img2.shape[1]) / 2)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            new_img[x + left_x_min_coordinate, y + left_y_min_coordinate] = img[x, y]
    return new_img


def blend(img1, img2):
    print('blending')
    blended_image = np.zeros(BLENDED_IMAGE_SIZE)
    img1_intensity = np.sum(img1, axis=2) / 3
    img2_intensity = np.sum(img2, axis=2) / 3
    for x in range(blended_image.shape[0]):
        for y in range(blended_image.shape[1]):
            if img1_intensity[x, y] > img2_intensity[x, y]:
                blended_image[x, y] = img1[x, y]
            else:
                blended_image[x, y] = img2[x, y]
    return blended_image


files = ['images/input/left-2.jpg',
         'images/input/left-1.jpg',
         'images/input/middle.jpg',
         'images/input/right-1.jpg',
         'images/input/right-2.jpg', ]
BLENDED_IMAGE_SIZE = [2000, 4000, 3]
NUMBER_OF_POINTS = [5, 12]
selected_points = pickle.load(open('tmp/selected_points.p', 'rb'))

# # of points, # of wrong points, normalization, gaussian_noise
options = [
    [5, 0, True, 0],
    [12, 0, True, 0],
    [12, 3, False, 0],
    [12, 3, True, 0],
    [12, 5, True, 0],
    [12, 0, True, 1],
    [12, 0, True, 5],
    [12, 0, True, 10],
    [12, 0, False, 1],
    [12, 0, False, 5],
    [12, 0, False, 10],
]

for option in options:
    im1Points = np.flip(selected_points[1][:option[0]], axis=1)
    im2Points = np.flip(selected_points[2][:option[0]], axis=1) + np.random.normal(loc=0, scale=option[3])
    im3Points = np.flip(selected_points[3][:option[0]], axis=1)
    # If there is wrong match
    if option[1] > 0:
        shuffle_indexes_1 = np.random.permutation(np.arange(option[0]))[:option[1]]
        shuffle_indexes_2 = np.random.permutation(shuffle_indexes_1)
        im2Points[shuffle_indexes_1] = im2Points[shuffle_indexes_2]
    homography_matrix1_2 = computeH(im1Points, im2Points, normalize=option[2])
    homography_matrix3_2 = computeH(im3Points, im2Points, normalize=option[2])

    img1 = plt.imread("images/input/left-1.jpg")
    img2 = plt.imread("images/input/middle.jpg")
    img3 = plt.imread("images/input/right-1.jpg")
    new_img1, new_img_corner_coordinates = warp(img1, homography_matrix1_2)
    new_img3, new_img_corner_coordinates = warp(img3, homography_matrix3_2)
    new_img2 = expand(img2).astype(np.int)
    blended = blend(new_img1, new_img2).astype(np.uint8)
    blended = blend(new_img3, blended).astype(np.uint8)
    # plt.figure()
    # plt.imshow(blended)
    # plt.show()
    plt.imsave(f'images/output/blended_images_3_points_{option[0]}_wrong_{option[1]}__normalized_{option[2]}_noise_{option[3]}.png', blended)
