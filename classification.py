import argparse
import cv2
import itertools
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from sklearn.cluster import KMeans
import utils





ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path of input image")
ap.add_argument("-t", "--threshold", required = False, type = int,
	help = "threshold for THRESH_BINARY")
args = vars(ap.parse_args())

custom_threshold = 40
if args["threshold"]:
	custom_threshold = args["threshold"]


model = load_model('model.h5')
value = 0.0
value_list = [0.01, 0.2, 2, 0.02, 0.5, 0.05] # TODO
count = 0
euro_labels = ['1ct', '20ct', '2E', '2ct', '50ct', '5ct'] # TODO

# load image
image = cv2.imread(args["image"])

# preprocess image
shifted = cv2.pyrMeanShiftFiltering(image, 10, 50)
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

# separate background from coins
thresh = cv2.threshold(gray, custom_threshold, 255,
	cv2.THRESH_BINARY)[1]

cv2.imshow("Input", image)
cv2.imshow("Shifted", shifted)
cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)

# loop all areas found
for label in np.unique(labels):
	# ignore background
	if label == 0:
		continue


	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(cnts, key=cv2.contourArea)

	# draw the border of the coin
	((x, y), r) = cv2.minEnclosingCircle(c)

	# remove noise, coin radius is always bigger than 30
	if r < 30:
		continue
	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)

    # locate coin in the image 
	from_x = int(x) - int(r)
	from_y = int(y) - int(r)

	to_x = int(x) + int(r)
	to_y = int(y) + int(r)

	if from_x < 0:
		from_x = 0
	if from_y < 0:
		from_y = 0
	if to_x >= image.shape[1]:
		to_x = image.shape[1] - 1
	if to_y >= image.shape[0]:
		to_y = image.shape[0] - 1

	# copy coin and find most dominant color
	coin = image[from_y:to_y, from_x:to_x, :]
	(red, green, blue) = get_dominant_color(coin)

	coin_features = np.array([r,red,green,blue]).reshape((1,4))
	result = model.predict(coin_features, verbose=0)

	# get best prediction and it's label
	m = max(result[0])
	idx = [i for i, j in enumerate(result[0]) if j == m][0]
	label_text = euro_labels[idx] #encoder.classes_

	# add value of predicted coin to total value and increase count
	value += value_list[idx]
	count += 1

	# label coin in the output image
	cv2.putText(image, label_text, (int(x) - 10, int(y)), #"#{}".format(label)
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# print results
value = float("{0:.2f}".format(value))
cv2.putText(image, "Value: " + str(value), (10, 20), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(image, "Coins: " + str(count), (10, 45), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# show the output image
cv2.imshow("Output", image)
#cv2.waitKey(0)