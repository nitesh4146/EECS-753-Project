import pandas as pd
import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt

def crop_scale(img):

    sky_crop = 530
    hood_crop = 900
    left_crop = 300
    right_crop = 1600
    
    processed_img = img[sky_crop:hood_crop, left_crop:right_crop, :]

    # processed_img = rescale(processed_img)
    processed_img = cv2.resize(processed_img, (200,66))

    # print(processed_img.shape)
    # cv2.imshow("processed_img", processed_img)
    # cv2.waitKey()

    return processed_img


def rescale(img):
    scale_percent = 60 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def equalize_data(data, labels):
    nbin = 25
    hist, bins = plot_hist(labels, nbin)

    # Finding mean of measurement data
    avg_value = np.mean(hist)
    print("Mean of Measurement: ", avg_value)

    # Randomly sampling data to fit within the mean value
    keep_prob = []
    for i in range(nbin):
        if hist[i] <= avg_value:
            keep_prob.append(1)
        else:
            keep_prob.append(avg_value/hist[i])

    remove_ind = []
    for i in range(len(labels)):
        for j in range(nbin):
            if labels[i] > bins[j] and labels[i] <= bins[j+1]:
                if random.random() < (1-keep_prob[j]):
                    remove_ind.append(i)

    # Removing images and measurements above mean value
    df1= pd.DataFrame(list(zip(data, labels)))
    df1.drop(remove_ind, inplace=True)

    eq_data = df1[0].tolist()
    eq_labels = df1[1].tolist()

    plot_hist(eq_labels)

    # print(new_images)
    # print(eq_labels.shape)
    return np.array(eq_data), np.array(eq_labels)


def plot_hist(labels, nbin=25):
    hist, bins = np.histogram(labels, nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

    return hist, bins


# crop_scale(cv2.imread("/home/slickmind/lanefollowing/ros2_ws/src/lane_following/bumblebee/images/center-2021-05-05T19:45:42.285457.jpg"))
