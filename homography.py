import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

selected_points = pickle.load(open('tmp/selected_points.p', 'rb'))
im1Points = np.array(selected_points[0])
im2Points = np.array(selected_points[1])


def normalize(points):
    mean_of_images = np.mean(points, 1)
    dist_of_points = np.sum(np.sqrt(np.sum(np.square(points), axis=2)), axis=1)
    print(dist_of_points)
    print(mean_of_images)
    for i in range(len(points)):
        points[i] = points[i] - mean_of_images[i]
    return points / dist_of_points * np.sqrt(2)


def normalization_matrix(points):
    nm = np.diag(np.ones(3))
    mean_of_points = np.mean(points, 0)
    dist_of_points = np.sum(np.square(np.sum(np.square(points), 1)))
    nm[0, 2] -= mean_of_points[0]
    nm[1, 2] -= mean_of_points[1]
    nm[:2, :] /= dist_of_points
    return nm


def create_A_matrix_columns(im1point, im2point):
    col1 = np.array(
        [0, 0, 0, -im1point[0], -im1point[1], -1, im2point[1] * im1point[0], im1point[1] * im2point[1], im2point[1]])
    col2 = np.array(
        [im1point[0], im1point[1], 1, 0, 0, 0, -im2point[0] * im1point[0], -im2point[0] * im1point[1], -im2point[0]])
    cols = np.stack((col1, col2))
    return cols


def create_A_matrix(im1Points, im2Points):
    A = []
    for i in range(len(selected_points[0])):
        cols = create_A_matrix_columns(im1Points[i], im2Points[i])
        A.append(cols)

    return np.vstack(A)


def computeH(im1Points, im2Points):
    im1_normalization_matrix = normalization_matrix(im1Points)
    im2_normalization_matrix = normalization_matrix(im2Points)
    im1Points_normalized = (im1_normalization_matrix @ np.insert(im1Points, 2, 1, axis=1).T).T
    im2Points_normalized = (im2_normalization_matrix @ np.insert(im2Points, 2, 1, axis=1).T).T
    u, s, vh = np.linalg.svd(create_A_matrix(im1Points_normalized, im2Points_normalized))
    normalized_homography_matrix = np.reshape(vh[-1, :] / vh[-1, -1], (3, 3))
    return np.linalg.inv(im2_normalization_matrix) @ normalized_homography_matrix @ im1_normalization_matrix


def warp(image, H):
    new_coordinates = np.zeros((image.shape[0], image.shape[1], 2))
    min_x, min_y, max_x, max_y = 0, 0, 0, 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_coordinate = (H @ np.array([x, y, 1]).T).astype(np.int)
            min_x = min(min_x, new_coordinate[0])
            max_x = max(max_x, new_coordinate[0])
            new_coordinates[x, y, :] = new_coordinate[:2]

    new_img = np.zeros((max_x - min_x + 1, image.shape[1], 3))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_coordinate = (new_coordinates[x, y] - [min_x, min_y]).astype(np.int)
            new_img[new_coordinate[0], new_coordinate[1]] = image[x, y]

    # if 0 <= new_coordinate[0] < image.shape[0] and 0 <= new_coordinate[1] < image.shape[1]:
    #     new_image[int(new_coordinate[0]), int(new_coordinate[1])] = image[x, y]

    return new_img.astype(np.uint8), min_x, max_x


def interpolate(new_image):
    mask = np.array(np.sum(new_image == 0, axis=2), dtype=np.uint8)
    interpolated_image = cv2.inpaint(src=new_image, inpaintMask=mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    return interpolated_image


def blend(img1, img2, min_x, max_x):
    blended_image = np.zeros((max(img2.shape[0] - min_x, max_x - min_x), max(img2.shape[1] - min_y, max_y - min_y), 3))
    for x in range(blended_image.shape[0]):
        for y in range(blended_image.shape[1]):
            if img2.shape[1] - min_x > x > -min_x and img2.shape[1] - min_y > y > -min_y:
                blended_image[x, y, :] = img2[x + min_x, y + min_y, :]
            elif x < max_x - min_x - 1 and y < max_y - min_y - 1:
                blended_image[x, y, :] = img1[x, y, :]

    return blended_image


homography_matrix = computeH(im1Points, im2Points)

img1 = plt.imread("images/input/left-1.jpg")
img2 = plt.imread("images/input/middle.jpg")

new_img1, (min_x, min_y), (max_x, max_y) = warp(img1, homography_matrix)

plt.imshow(new_img1)
plt.show()
new_img1 = interpolate(new_img1)
plt.imshow(new_img1)
plt.show()
blended = blend(new_img1, img2, min_x, min_y, max_x, max_y)
plt.imshow(blended)
plt.show()
