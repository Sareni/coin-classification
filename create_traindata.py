import argparse
import glob

import cv2
import numpy as np

def printOnImage(image, text, xposition, yposition, color):
    cv2.putText(image, text,
                (xposition, yposition), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, thickness=2, lineType=cv2.LINE_8)

def get_testdata_from_file(image, nr, output_flag):
    d = 1024 / image.shape[1]
    dim = (1024, int(image.shape[0] * d))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                               param1=170, param2=100, minRadius=8, maxRadius=120)

    count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, d) in circles:
            count += 1

            coin = image[y - d:y + d, x - d:x + d]

            # create a mask of the coin
            m = np.zeros(coin.shape[:2], dtype="uint8")
            w = int(coin.shape[1] / 2)
            h = int(coin.shape[0] / 2)
            cv2.circle(m, (w, h), d, (255), -1)
            maskedCoin = cv2.bitwise_and(coin, coin, mask=m)
            # write to a separate folder
            cv2.imwrite("traindata\\train{}{}.png".format(nr,count), maskedCoin)
            cv2.circle(output, (x, y), d, (0, 0, 0), 2)
    if output_flag & count < 10:
        d = 768 / output.shape[1]
        dim = (768, int(output.shape[0] * d))
        output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", output)
        cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dev", required=False, help="developer mode", default=False, action='store_true')
args = vars(ap.parse_args())

filenames = glob.glob("raw_testdata/*.*")
filenames.sort()
images = [cv2.imread(img) for img in filenames]

count = 0
for img in images:
    get_testdata_from_file(img,count,args["dev"])
    count += 1



