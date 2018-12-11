import numpy as np
import os
import pickle

from scipy.cluster.vq import *


class SpatialPyramidHistogram(object):

    def normalize_histogram(self, histogram):
        norm = np.linalg.norm(histogram)

        if norm > 1.0:
            histogram /= float(norm)

        return histogram

    def build_spatial_pyramid_histogram(self, img_features, vocabulary, size):
        width = img_features['width']
        height = img_features['height']

        width_step = int(width / 4)
        height_step = int(height / 4)

        descriptors = img_features['descriptors']

        histogram_level_two = np.zeros((16, vocabulary.shape[0]))
        for descriptor in descriptors:
            x = descriptor['x']
            y = descriptor['y']
            boundary_index = int(x / width_step) + int(y / height_step) * 4

            feature = descriptor['vector']
            shape = feature.shape[0]
            feature = feature.reshape(1, shape)

            codes, distance = vq(feature, vocabulary)
            histogram_level_two[boundary_index][codes[0]] += 1

        histogram_level_one = np.zeros((4, size))
        histogram_level_one[0] = histogram_level_two[0] + histogram_level_two[1] + histogram_level_two[4] + \
                                 histogram_level_two[5]
        histogram_level_one[1] = histogram_level_two[2] + histogram_level_two[3] + histogram_level_two[6] + \
                                 histogram_level_two[7]
        histogram_level_one[2] = histogram_level_two[8] + histogram_level_two[9] + histogram_level_two[12] + \
                                 histogram_level_two[13]
        histogram_level_one[3] = histogram_level_two[10] + histogram_level_two[11] + histogram_level_two[14] + \
                                 histogram_level_two[15]

        histogram_level_zero = histogram_level_one[0] + histogram_level_one[1] + histogram_level_one[2] + \
                               histogram_level_one[3]

        tempZero = histogram_level_zero.flatten() * 0.25
        tempOne = histogram_level_one.flatten() * 0.25
        tempTwo = histogram_level_two.flatten() * 0.5
        result = np.concatenate((tempZero, tempOne, tempTwo))
        normalized_result = self.normalize_histogram(result)

        return normalized_result

    def build_histogram(self, feature):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        vocabulary_path = os.path.join(BASE_DIR, '../machine-learning-models/vocabulary.sav')

        file = open(vocabulary_path, "rb")
        vocabulary = pickle.load(file)
        size = vocabulary.shape[0]

        img_histogram = {}
        img_histogram['file_name'] = feature['file_name']
        img_histogram['histogram'] = self.build_spatial_pyramid_histogram(feature, vocabulary, size)

        return img_histogram
