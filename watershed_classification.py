import argparse
import cv2
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from sklearn.cluster import KMeans
import utils
import os

# create copy of image with specific rotation angle
def create_copy(path, num, angle):
	input_img = cv2.imread(path)

	(h, w) = input_img.shape[:2]
	center = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	output_img = cv2.warpAffine(input_img, M, (h, w))

	k = path.rfind("_")
	orig_num = int(path[k+1])
	new_path = path[:k] + "_" + str(orig_num + num) + ".png"

	cv2.imwrite(new_path, output_img)

# resize image to a defined size
def normalize_images(path):
	input_img = cv2.imread(path)
	output_img = cv2.resize(input_img, (120, 120)) 
	cv2.imwrite(path, output_img)

# get three domiant colors of the image by KMeans
def get_dominant_color(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	clt = KMeans(n_clusters = 3)
	clt.fit(image)
	hist = utils.centroid_histogram(clt)

	m = max(hist)
	idx = [i for i, j in enumerate(hist) if j == m]

	return tuple(list(itertools.chain.from_iterable(clt.cluster_centers_[idx].astype("uint8").tolist())))

def get_image_area_labeled(image, sp, verbose, custom_threshold):

    # preprocess image
    shifted = cv2.pyrMeanShiftFiltering(image, sp, 50)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    # separate background from coins
    thresh = cv2.threshold(gray, custom_threshold, 255,
        cv2.THRESH_BINARY)[1]

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

    if verbose > 0:
        cv2.imshow("Input", image)
        cv2.imshow("Shifted", shifted)
        cv2.imshow("Thresh", thresh)


    return (labels, gray.shape)

def getAttributesOfCoin(image):
    (labels, mask_shape) = get_image_area_labeled(image, 3, 0, 40)
    max_r = 0
    max_x = 0
    max_y = 0

    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(mask_shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > max_r:
            max_r = r


    (red, green, blue) = get_dominant_color(image.copy())
    return (max_r, red, green, blue)

# add new row to train/test data set, mult_factor can be used to create multiple entries per image
def add_new_row (df, filename, path, mult_factor):
	class_label = filename.split("_")[0]
	image = cv2.imread(path)
	(radius, red, green, blue) = getAttributesOfCoin(image)

	# TODO remove background dominated images // threshold sum 80
	# TODO set index
	for _ in range(0, int(mult_factor)):
		row = pd.DataFrame([[class_label, radius, red, green, blue, path]], columns=['Class Label', 'Radius', 'Red', 'Green', 'Blue', 'Path'])
		df = df.append(row)
	return df

# create a simple NN
def create_model():
	print('Creating Model...')
	model = Sequential()
	model.add(Dense(100, input_dim=4, activation='tanh'))
	model.add(Dense(6))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Done!')
	return model

def train_model(model, df):
    print('Train Model...')

    seed = 7
    np.random.seed(seed)

    X = df.as_matrix(columns=['Radius', 'Red', 'Green', 'Blue'])
    Y = df.as_matrix(columns=['Class Label'])

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)

    model.fit(X, dummy_y, batch_size=5, nb_epoch=50, verbose=1, shuffle=True)
    #score = model.evaluate(X_test, Y_test, verbose=0)
    #print(score)
    model.save('model.h5')
    print('Done!')
    return (model, encoder.classes_)


def create_data(mult_factor):
    df = pd.DataFrame(columns=['Class Label', 'Radius', 'Red', 'Green', 'Blue', 'Path'])

    IMAGE_DIRECTORY = './train_images/'
    directory = os.fsencode(IMAGE_DIRECTORY)

    # needed for testing
    #for file in os.listdir(directory):
        #filename = os.fsdecode(file)
        #if filename.endswith(".png"):
            #normalize_images(os.path.join(IMAGE_DIRECTORY, filename))
            #create_copy(os.path.join(IMAGE_DIRECTORY, filename), 1, 90)
            #create_copy(os.path.join(IMAGE_DIRECTORY, filename),  2, 180)
            #create_copy(os.path.join(IMAGE_DIRECTORY, filename), 3, 270)
	
    print('Creating Training Data ...')

    # loop all images in the image data folder
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            df = add_new_row(df, filename, os.path.join(IMAGE_DIRECTORY, filename), mult_factor)

    print('Done!')
    return df

def main_function():
    print('Initializing ...')
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path of input image")
    ap.add_argument("-t", "--threshold", required = False, type = int,
        help = "threshold for THRESH_BINARY")
    ap.add_argument("-f", "--factor", required=False,
	help="data multiplication factor")
    args = vars(ap.parse_args())

    custom_threshold = 40
    if args["threshold"]:
        custom_threshold = args["threshold"]

    mult_factor = 1
    if args["factor"]:
	    mult_factor = args["factor"]

    value_dict = {
        '1ct': 0.01,
        '2ct': 0.02,
        '5ct': 0.05,
        '10ct': 0.1,
        '20ct': 0.2,
        '50ct': 0.5,
        '1E': 1.0,
        '2E': 2.0
    }

    value = 0.0
    count = 0

    print('Done!')
    df = create_data(mult_factor)
    df.to_csv("test.csv")
    model = create_model()
    (model, classification_labels) = train_model(model, df)

    print('Classify Image ...')
    # load image
    image = cv2.imread(args["image"])

    (labels, mask_shape) = get_image_area_labeled(image, 10, 1, custom_threshold)

    # loop all areas found
    for label in np.unique(labels):
        # ignore background
        if label == 0:
            continue

        mask = np.zeros(mask_shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw the border of the coin
        ((x, y), r) = cv2.minEnclosingCircle(c)

        # remove noise, coin radius is always bigger than 20
        if r < 20:
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

        # select coin and find most dominant color
        coin = image[from_y:to_y, from_x:to_x, :]
        (red, green, blue) = get_dominant_color(coin)

        coin_features = np.array([r, red, green, blue]).reshape((1,4))
        result = model.predict(coin_features, verbose=0)

        # get best prediction and it's label
        m = max(result[0])
        idx = [i for i, j in enumerate(result[0]) if j == m][0]
        label_text = classification_labels[idx] 

        # add value of predicted coin to total value and increase count
        value += value_dict[label_text]
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

    print('Done!')

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


main_function()
