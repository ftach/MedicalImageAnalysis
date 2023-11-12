import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import time
import skimage.io as skio
import tensorflow.keras.metrics as tf_metrics
from skimage.color import rgb2gray
from skimage.util import invert
import sys
from sklearn.metrics import accuracy_score

# TO MODIFY     # TO MODIFY     # TO MODIFY     # TO MODIFY     # TO MODIFY
seeds = [(15, 15), (250, 200), (75, 75), (232, 125),
         (240, 60), (50, 200)]  # atopic dermatitis


# seeds = [(320, 200)]  # ultrasound


class Stack():
    def __init__(self):
        self.item = []
        self.obj = []

    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []


class regionGrow():

    def __init__(self, im_path, th):
        self.readImage(im_path)
        self.h, self.w, _ = self.im.shape
        self.passedBy = np.zeros((self.h, self.w), np.double)
        self.currentRegion = 0
        self.iterations = 0
        self.SEGS = np.zeros((self.h, self.w, 3), dtype='uint8')
        self.stack = Stack()
        self.thresh = float(th)

    def readImage(self, img_path):
        self.im = cv2.imread(img_path, 1).astype('int')

    def getNeighbour(self, x0, y0):
        return [
            (x, y)
            for i, j in itertools.product((-1, 0, 1), repeat=2)
            if (i, j) != (0, 0) and self.boundaries(x := x0 + i, y := y0 + j)
        ]

    def ApplyRegionGrow(self, seeds, cv_display=True):
        temp = []
        for i in seeds:
            temp.append(i)
            temp.extend(self.getNeighbour(*i))
        seeds = temp
        for i in (seeds):
            x0, y0 = *i,

            if self.passedBy[x0, y0] == 0 and np.all(self.im[x0, y0] > 0):
                self.currentRegion += 1
                self.passedBy[x0, y0] = self.currentRegion
                self.stack.push((x0, y0))

                while not self.stack.isEmpty():
                    x, y = self.stack.pop()
                    self.BFS(x, y)
                    self.iterations += 1
                if (self.PassedAll()):
                    break
                count = np.count_nonzero(self.passedBy == self.currentRegion)
                if (count < 8 * 8):
                    x0, y0 = self.reset_region(x0, y0)

        if cv_display:
            [self.color_pixel(i, j) for i, j in itertools.product(
                range(self.h), range(self.w))]
            # self.display()

        return self.SEGS

    def reset_region(self, x0, y0):
        self.passedBy[self.passedBy == self.currentRegion] = 0
        self.currentRegion -= 1
        return x0 - 1, y0 - 1

    def BFS(self, x0, y0):
        regionNum = self.passedBy[x0, y0]
        elems = [
            np.mean(self.im[x0, y0])
        ]
        var = self.thresh
        neighbours = self.getNeighbour(x0, y0)
        for x, y in neighbours:
            if self.passedBy[x, y] == 0 and self.distance(x, y, x0, y0) < var:
                if (self.PassedAll()):
                    break
                self.passedBy[x, y] = regionNum
                self.stack.push((x, y))
                elems.append(np.mean(self.im[x, y]))
                var = np.var(elems)
            var = max(var, self.thresh)

    def color_pixel(self, i, j):
        val = self.passedBy[i][j]
        self.SEGS[i][j] = (255, 255, 255) if (
            val == 0) else (val*35, val*90, val*30)

    def display(self):
        cv2.imshow("", self.SEGS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def PassedAll(self, max_iteration=200000):

        return self.iterations > max_iteration or np.all(self.passedBy > 0)

    def boundaries(self, x, y):
        return 0 <= x < self.h and 0 <= y < self.w

    def distance(self, x, y, x0, y0):
        return np.linalg.norm(self.im[x0, y0] - self.im[x, y])


def get_seeds(seeds):
    return [(y, x) for x, y in seeds]


def main(seeds=[(320, 200)], mask_comparison=False):
    if len(sys.argv) < 3:
        print("Please specify an image path and a threshold value")
        exit(1)
    else:
        img_path = sys.argv[1]
        threshold = sys.argv[2]
        seeds = get_seeds(seeds)

        start_time = time.time()
        exemple = regionGrow(img_path, threshold)
        segmented_image = exemple.ApplyRegionGrow(seeds)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        if mask_comparison:
            if len(sys.argv) < 4:
                print("Please specify a mask image path")
                exit(1)
            else:
                segmented_image = rgb2gray(segmented_image)
                segmented_image = invert(segmented_image)
                print(np.min(segmented_image), np.max(segmented_image))
                segmented_image = segmented_image / np.max(segmented_image)
                mask_path = sys.argv[3]
                mask = skio.imread(mask_path, as_gray=False)
                mask = mask / np.max(mask)

                m = tf_metrics.MeanIoU(num_classes=2)
                m.update_state(mask, segmented_image)

                print("IoU:", m.result().numpy())
                Accuracy = accuracy_score(mask, segmented_image)
                print(f'Accuracy: {Accuracy}')

        plt.figure()
        plt.imshow(segmented_image, cmap='gray'), plt.axis('off')
        # plt.imsave('rg_us_segmented.png', segmented_image, cmap='gray')
        plt.show()


if __name__ == "__main__":
    main(seeds=seeds, mask_comparison=True)
