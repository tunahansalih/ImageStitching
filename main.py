import cv2
import matplotlib.pyplot as plt
import pickle

selected_points = []

img = plt.imread("images/input/left-1.jpg")
plt.imshow(img)
selected_points.append(plt.ginput(8, show_clicks=True))
print('clicked', selected_points)
plt.close()

img = plt.imread("images/input/middle.jpg")
plt.imshow(img)
selected_points.append(plt.ginput(8, show_clicks=True))
print('clicked', selected_points)
plt.close()

pickle.dump(selected_points, open('tmp/selected_points.p', 'wb'))
