import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

NUM_OF_INTEREST_POINTS = 12
files = ['images/input/left-2.jpg',
         'images/input/left-1.jpg',
         'images/input/middle.jpg',
         'images/input/right-1.jpg',
         'images/input/right-2.jpg', ]
selected_points = np.zeros((len(files), NUM_OF_INTEREST_POINTS, 2))
images = []

for f in files:
    img = plt.imread(f)
    images.append(img)


for i in range(12):
    for j, img in enumerate(images):
        plt.imshow(img)
        clicked = plt.ginput(1, show_clicks=True, timeout=0)
        selected_points[j,i,:] = np.array(clicked)
        print(clicked)
        plt.close()


pickle.dump(selected_points, open('tmp/selected_points.p', 'wb'))