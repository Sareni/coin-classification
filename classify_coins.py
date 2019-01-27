import argparse
import glob
import math

import matplotlib.pyplot as plt

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# define Enum class
class Enum(tuple): __getattr__ = tuple.index

# print text on the image
def printOnImage(image, text, xposition, yposition, color):
    cv2.putText(image, text,
                (xposition, yposition), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, thickness=2, lineType=cv2.LINE_8)

# train
modelFileName = "MLPmodel.sav"

# calculate the feature histogram (only for the circle, no background)
def calcHist(img):
    # create mask
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 60, 255, -1)

    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def calcHistFromFile(file):
    img = cv2.imread(file)
    return calcHist(img)

# training data classes
Material = Enum(('Copper', 'Brass', 'Euro1', 'Euro2'))

def trainModel():
    sample_images_copper = glob.glob("sample_images/copper/*")
    sample_images_brass = glob.glob("sample_images/brass/*")
    sample_images_euro1 = glob.glob("sample_images/euro1/*")
    sample_images_euro2 = glob.glob("sample_images/euro2/*")

    X = []
    Y = []

    for i in sample_images_copper:
        X.append(calcHistFromFile(i))
        Y.append(Material.Copper)
    for i in sample_images_brass:
        X.append(calcHistFromFile(i))
        Y.append(Material.Brass)
    for i in sample_images_euro1:
        X.append(calcHistFromFile(i))
        Y.append(Material.Euro1)
    for i in sample_images_euro2:
        X.append(calcHistFromFile(i))
        Y.append(Material.Euro2)

    # Multi-layer Perceptron
    seed = 7
    classifier = MLPClassifier(solver="lbfgs")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=seed)

    # train and score classifier
    classifier.fit(X_train, Y_train)
    score = int(classifier.score(X_test, Y_test) * 100)
    print("Classifier mean accuracy: ", score)

    # plot
    # plt.plot(classifier.loss_)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    #
    # plt.show()

    # save model
    tuble_objects = (classifier, X_train, Y_train, score)
    joblib.dump(tuble_objects, modelFileName)
    return classifier

# file input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
ap.add_argument("-d", "--dev", required=False, help="developer mode", default=False, action='store_true')
ap.add_argument("-t", "--train", required=False, help="train model", default=False, action='store_true')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
if image is None:
    print("Could not open or find the image: ", args["image"])
    exit(0)
#image = cv2.imread("input.jpg")

if args["train"]:
    classifier = trainModel()
else:
    classifier, X_train, Y_train, score = joblib.load(modelFileName)
    if classifier is None:
        print("Could not load model")
        exit(0)

# Preprocessing
d = 1024 / image.shape[1]
dim = (1024, int(image.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# create a copy of the image
output = image.copy()

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# improve contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Gaussian blurring,
# we use a 7x7 kernel and let OpenCV detect sigma
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# circles: A vector that stores x, y, r for each detected circle.
# src_gray: Input image (grayscale)
# CV_HOUGH_GRADIENT: Defines the detection method.
# dp = 2.2: The inverse ratio of resolution
# min_dist = 100: Minimum distance between detected centers
# param_1 = 200: Upper threshold for the internal Canny edge detector
# param_2 = 100*: Threshold for center detection.
# min_radius = 50: Minimum radius to be detected.
# max_radius = 120: Maximum radius to be detected.
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=50,
                           param1=200, param2=100, minRadius=10, maxRadius=120)

def prediction(coin):
    hist = calcHist(coin)
    s = classifier.predict([hist])
    return Material[int(s)]

diameter = []
materials = []
coordinates = []

count = 0
if circles is not None:
    # append radius to list of diameters (we don't bother to multiply by 2)
    for (x, Y, r) in circles[0, :]:
        diameter.append(r)

    # convert coordinates and radii to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over coordinates and radii of the circles
    for (x, Y, d) in circles:
        count += 1
        coordinates.append((x, Y))
        coin = image[Y - d:Y + d, x - d:x + d]

        material = prediction(coin)
        materials.append(material)

        # draw contour and material
        cv2.circle(output, (x, Y), d, (0, 0, 0), 2)
        printOnImage(output, material, x - 40, Y, (0, 0, 0))

# get biggest coin
biggest = max(diameter)
i = diameter.index(biggest)

if materials[i] == "Euro2":
    diameter = [x / biggest * 25.75 for x in diameter]
    scaledTo = "Scaled to 2 Euro"
elif materials[i] == "Brass":
    diameter = [x / biggest * 24.25 for x in diameter]
    scaledTo = "Scaled to 50 Cent"
elif materials[i] == "Euro1":
    diameter = [x / biggest * 23.25 for x in diameter]
    scaledTo = "Scaled to 1 Euro"
elif materials[i] == "Copper":
    diameter = [x / biggest * 21.25 for x in diameter]
    scaledTo = "Scaled to 5 Cent"
else:
    scaledTo = "unable to scale..."
print(scaledTo)

i = 0
total = 0
while i < len(diameter):
    d = diameter[i]
    m = materials[i]
    (x, Y) = coordinates[i]
    t = "Unknown"

    # compare to known diameters with some margin for error
    if math.isclose(d, 25.75, abs_tol=2.5) and m == "Euro2":
        t = "2 Euro" # 1.25
        total += 200
    elif math.isclose(d, 23.25, abs_tol=2.5) and m == "Euro1":
        t = "1 Euro"
        total += 100
    elif math.isclose(d, 19.75, abs_tol=1.25) and m == "Brass":
        t = "10 Cent"
        total += 10
    elif math.isclose(d, 22.25, abs_tol=1.0) and m == "Brass":
        t = "20 Cent"
        total += 20
    elif math.isclose(d, 24.25, abs_tol=2.5) and m == "Brass":
        t = "50 Cent"
        total += 50
    elif math.isclose(d, 16.25, abs_tol=1.25) and m == "Copper":
        t = "1 Cent"
        total += 1
    elif math.isclose(d, 18.75, abs_tol=1.25) and m == "Copper":
        t = "2 Cent"
        total += 2
    elif math.isclose(d, 21.25, abs_tol=2.5) and m == "Copper":
        t = "5 Cent"
        total += 5

    printOnImage(output, t, x - 40, Y + 22, (255, 255, 255))
    i += 1

d = 768 / output.shape[1]
dim = (768, int(output.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
blurred = cv2.resize(blurred, dim, interpolation=cv2.INTER_AREA)
output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

# write summary on output image
printOnImage(output, "EUR {} ({} Coins)".format(total / 100, count), 5, 20, (0,0,0))

# show steps in dev mode
if args["dev"]:
    cv2.imshow("Original", image)
    cv2.imshow("Contrast", gray)
    cv2.imshow("Gauss", blurred)

# show output and wait for key to terminate program
cv2.imshow("Output",output)
cv2.waitKey(0)