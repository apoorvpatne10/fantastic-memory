import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import os
import imutils
import dlib # run "pip install dlib"
import cv2 # run "pip install opencv-python"

from scipy import misc # run "pip install pillow"
from imutils import face_utils

RECTANGLE_LENGTH = 90

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
    	coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def crop_and_save_image(img, img_path, write_img_path, img_name):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # load the input image, resize it, and convert it to grayscale

    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if len(rects) > 1:
    	print( "ERROR: more than one face detected")
    	return
    if len(rects) < 1:
    	print( "ERROR: no faces detected")
    	return

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        name, i, j = 'mouth', 48, 68
        # clone = gray.copy()

        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        # w = RECTANGLE_LENGTH
        # h = RECTANGLE_LENGTH
        roi = gray[y:y+h, x:x+w]
        roi = imutils.resize(roi, width = 250, inter=cv2.INTER_CUBIC)
        print('cropped/' + write_img_path)
        cv2.imwrite('cropped/' + write_img_path, roi)

people_small = ['F01','F02']
people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
data_types = ['phrases', 'words']
folder_enum = ['01','02','03','04','05','06','07','08','09','10',]

VALIDATION_SPLIT = ['F07']
TEST_SPLIT = ['F11']

X_train = None
y_train = None

X_val = None
y_val = None

X_test = None
y_test = None

if not os.path.exists('cropped'):
    os.mkdir('cropped')

for person_ID in people:
    if not os.path.exists('cropped/' + person_ID ):
        os.mkdir('cropped/' + person_ID)

    for data_type in data_types:
        if not os.path.exists('cropped/' + person_ID + '/' + data_type):
            os.mkdir('cropped/' + person_ID + '/' + data_type)

        for phrase_ID in folder_enum:
            if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID):
                # F01/phrases/01
                os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID)

            for instance_ID in folder_enum:
                # F01/phrases/01/01
                directory = 'dataset/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'
                dir_temp = person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'
                # print(directory)
                filelist = os.listdir(directory)
                if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID):
                    os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID)

                    for img_name in filelist:
                        if img_name.startswith('color'):
                            image = misc.imread(directory + '' + img_name)
                            crop_and_save_image(image, directory + '' + img_name,
                                                dir_temp + '' + img_name, img_name)
