import itertools
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import tensorflow.keras.metrics as tf_metrics
from sklearn.metrics import accuracy_score
import cv2
from skimage.morphology import flood_fill
from skimage.util import invert
import pandas as pd
import math

noisy = skio.imread('malignant (17).png', as_gray=True)
mask = skio.imread('malignant (17)_mask.png', as_gray=True)


def disp_iou_accu(img, mask):
    if math.isnan(float(np.min(img))) == True or math.isnan(float(np.max(img))) == True:
        print("done")
        return 0, 0
    else:
        img = (img - np.min(img)) * (1.0 / (np.max(img) - np.min(img)))
        mask = mask / np.max(mask)
        m = tf_metrics.MeanIoU(num_classes=2)
        print(np.min(img), np.max(img))
        m.update_state(mask, img)
        iou = m.result().numpy()
        acc = accuracy_score(mask, img)
        return iou, acc


def snake_algo(alpha=0.015, beta=20, gamma=0.001, w_line=0.5, w_edge=-1, sigma=3):
    denoised = gaussian(noisy, sigma, preserve_range=False)

    s = np.linspace(0, 2*np.pi, 400)
    r = 180 + 110*np.sin(s)
    c = 370 + 150*np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(denoised, init, alpha=alpha,
                           beta=beta, gamma=gamma, w_line=w_line, w_edge=w_edge)
    return snake


def fill_snake(img_shape, snake, fill_seed=(320, 200)):
    filled_img = np.zeros(img_shape)
    filled_img[snake[:, 0].astype(int), snake[:, 1].astype(int)] = 1
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(filled_img, kernel, iterations=2)
    filled_img = flood_fill(dilated_img, fill_seed, 1)
    filled_img = invert(filled_img)

    return filled_img


# Define the parameter values to test
alpha_values = [0.015, 0.01, 0.02]
beta_values = [10, 20, 30]
gamma_values = [0.001, 0.01, 0.005]
w_line_values = [-1, 0.5, 0]
w_edge_values = [-1, 0, 1]
sigma_values = [1, 3]


# Generate all combinations of parameter values
all_combinations = itertools.product(
    alpha_values, beta_values, gamma_values, w_line_values, w_edge_values, sigma_values)

results = pd.DataFrame(
    columns=['alpha', 'beta', 'gamma', 'w_line', 'w_edge', 'sigma', 'iou', 'acc'])
k = 0
for combination in all_combinations:
    k += 1
    print(k)
    alpha, beta, gamma, w_line, w_edge, sigma = combination
    combination = list(combination)

    snake = snake_algo(alpha, beta, gamma, w_line, w_edge, sigma)
    segmented = fill_snake(noisy.shape, snake)
    print(combination)
    iou, acc = disp_iou_accu(segmented, mask)
    combination.append(iou)
    combination.append(acc)
    results.loc[len(results)] = combination


results.to_csv('results_snakesearch.csv')
