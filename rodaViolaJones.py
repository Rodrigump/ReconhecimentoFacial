from __future__ import print_function
import cv2 as cv
import argparse

import skimage.io, skimage.color
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

def detectAndDisplay(frame,face_cascade):
    # retirado de https://docs.opencv.org/trunk/db/d28/tutorial_cascade_classifier.html
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

    #desenha circulo na face
    # for (x,y,w,h) in faces:
    #     center = (x + w//2, y + h//2)
    #     frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    #     faceROI = frame_gray[y:y+h,x:x+w]
    #     #-- In each face, detect eyes
    #     # eyes = eyes_cascade.detectMultiScale(faceROI)
    #     # for (x2,y2,w2,h2) in eyes:
    #     #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
    #     #     radius = int(round((w2 + h2)*0.25))
    #     #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    #desenha retangulo
    for (column, row, width, height) in faces:
        # print("faces " + str(nume) )
        cv.rectangle(
            frame,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2
        )
        # nume = nume + 1

    cv.imshow('Capture - Face detection', frame)

def rodaViolaJ(caminhoImagem):
    # testes do site: https://realpython.com/traditional-face-detection-python/
    # Read image from your local file system
    original_image = cv.imread(caminhoImagem)

    #teste canais imagem
    print("teste")
    # print(original_image.shape)

    # teste exibir imagem
    # cv.imshow('image',original_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Convert color image to grayscale for Viola-Jones
    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    # Load the classifier and create a cascade object for face detection
    # *verificar se tem como melhorar o endereço
    # face_cascade = cv.CascadeClassifier('C:\\Users\\pc\PycharmProjects\\EP2_IA_CLASSIFICATION\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')

    # detected_faces = face_cascade.detectMultiScale(grayscale_image)

    # nume = 1
    # for (column, row, width, height) in detected_faces:
    #     # print("faces " + str(nume) )
    #     cv.rectangle(
    #         original_image,
    #         (column, row),
    #         (column + width, row + height),
    #         (0, 255, 0),
    #         2
    #     )
    #     # nume = nume + 1
    # # ================================

    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default=cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default=cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    # parser.add_argument('--camera', help='Camera devide number.', type=int, default=0)
    args = parser.parse_args()

    face_cascade_name = args.face_cascade
    eyes_cascade_name = args.eyes_cascade

    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    # fd, hog_image = hog(original_image, orientations=8, pixels_per_cell=(16, 16),
    #                     cells_per_block=(1, 1), visualize=True, multichannel=True)
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)


    #-- 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    detectAndDisplay(original_image,face_cascade)

    rodaHOG(original_image)

    # cv.imshow('Image', original_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# # exibe imagem
#     ax1.axis('off')
#     ax1.imshow(original_image, cmap=plt.cm.gray)
#     ax1.set_title('Input image')
#
#     # Rescale histogram for better display
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
#     ax2.axis('off')
#     ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#     ax2.set_title('Histogram of Oriented Gradients')
#     plt.show()

    # ================================

def rodaHOG(original_image):

    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (2, 2)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    # regra para valores (altura/largura): (winSize - blockSize) % blockStride == 0

    # Cria descritor
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
    # descriptor = hog.compute(original_image, winStride=(64, 128), padding=(0, 0))

    array = np.array([])  # empty array for storing all the features
    # h = 128  # height of the image
    # w = 64  # width of the image

    # cv.imshow('Image', original_image)
    # img = cv.resize(original_image, (w, h), interpolation=cv.INTER_CUBIC)  # resize images
    # cv.imshow('Image', img)


    h = hog.compute(original_image, winStride=(64, 128), padding=(0, 0))  # storing HOG features as column vector
    # h = hog.compute(original_image)  # storing HOG features as column vector
    # # hog_image_rescaled = exposure.rescale_intensity(h, in_range=(0, 0.02))
    # # plt.figure(1, figsize=(3, 3))
    # # plt.imshow(h,cmap=plt.cm.gray)
    # # plt.show()
    # # print len(h)
    h_trans = h.transpose()  # transposing the column vector

    arrayHOG = np.vstack(h_trans)  # appending it to the array
    print ("HOG features of label 1")
    print (arrayHOG)


rodaViolaJ("teste1.jpg")


