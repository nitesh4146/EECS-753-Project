import os
from preprocess import crop_scale, equalize_data
import pandas as pd
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tkinter as tk
from tkinter import filedialog

# Set data directories
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = WORKING_DIR + "/maps"
DATA_CSV = "training_data.csv"
DATA_IMG = "img"
HDF5_DIR = WORKING_DIR + "/hdf5"
MODEL_DIR = os.path.join(WORKING_DIR, "model")

CAMERA_LIST = ["center", "right", "left"]
BATCH_SIZE = 2500


def data_to_hdf5():
    """
    Reads data (images and labels) from WORKING_DIR/maps/ and saves it to hdf5 files of size BATCH_SIZE each
    Args:
        None
    Returns:
        None
    """

    print("[info] Creaing new hdf5 files from map data...\n")

    print("Please make sure your data structure is as \n{}/maps/Map1/data[0,1,2...]/img".format(WORKING_DIR))
    if not os.path.exists(HDF5_DIR):
        os.makedirs(HDF5_DIR)

    # Get all maps
    all_maps = glob.glob(DATA_DIR + "/*")

    if not all_maps:
        print("Maps not found")
        exit(0)
    elif "Map" not in all_maps[0]:
        print("Incorrect directory structure")
        exit(0)

    maps_data = []
    maps_data += [glob.glob(map + "/*") for map in all_maps]

    images = []
    labels = []
    correction = [0.0, -0.02, 0.02]     # corrections w.r.t. [center right left]

    # Sample test
    sample_img_path = glob.glob(DATA_DIR + "/**/*.jpg", recursive=True)[0]
    img = cv2.imread(sample_img_path)
    # cv2.imwrite(WORKING_DIR, img)
    # plt.imshow(crop_scale(img), cmap="gray")
    # plt.show()

    batch = 0 
    batch_idx = 0

    for data in maps_data[0]:                               # Iterate over each map
        print("\n[info] Processing: ", data.split("/")[-2:])
        csv_df = pd.read_csv(data + "/" + DATA_CSV, header=None)

        for index, row in csv_df.iterrows():                # Iterate over data for each map
            # First column of DATA_CSV contains image timestamp
            img_time = row[0]
            # Second column of DATA_CSV contains steering value
            steer = row[1]

            for i in range(3):                              # Reading all CAMERA_LIST images
                img_path = data + "/" + DATA_IMG + "/" + \
                    CAMERA_LIST[i] + "-" + img_time + ".jpg"
                img = cv2.imread(img_path)
                img_processed = crop_scale(img)

                images.append(img_processed)
                labels.append(float(steer) + correction[i])

                images.append(np.fliplr(img_processed))
                labels.append(-(float(steer) + correction[i]))

                batch += 1

                if batch % BATCH_SIZE == 0:
                    h5_filename = os.path.join(HDF5_DIR, "batch-{}.h5".format(batch_idx))
                    with h5py.File(h5_filename, 'w') as hfile:
                        images_eq, labels_eq = equalize_data(images, labels)

                        hfile.create_dataset('images', data=images_eq)
                        hfile.create_dataset('labels', data=labels_eq)

                        print("Batch {} saved".format(batch_idx))
                        images = []
                        labels = []
                        batch_idx += 1
    
    # Save the remaining data 
    h5_filename = os.path.join(HDF5_DIR, "batch-{}.h5".format(batch_idx))
    with h5py.File(h5_filename, 'w') as hfile:
        images_eq, labels_eq = equalize_data(images, labels)

        hfile.create_dataset('images', data=images_eq)
        hfile.create_dataset('labels', data=labels_eq)

        print("Batch {} saved".format(batch_idx))
        images = []
        labels = []


def data_from_hdf5(filename):
    # print("[info] Loading data from saved hdf5...")

    with h5py.File(filename, 'r') as hfile:
        images = np.array(hfile.get('images'))
        labels = np.array(hfile.get('labels'))
    
    # print(images.shape)
    return images, labels


def dir_selector():
    root = tk.Tk()
    root.withdraw()

    DATA_DIR = filedialog.askdirectory(parent=root,initialdir="/",title='Please select your maps directory')
    print("DATA_DIR changed to", DATA_DIR)
