import cv2
import numpy as np
import os


class FeatureExtraction(object):

    def read_sift_from_vlfeat_file(self, file_name):
        f = np.loadtxt(file_name)
        return f[:, :4], f[:, 4:]

    def normalize_sift(self, descriptor):
        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)

        if norm > 1.0:
            descriptor /= float(norm)

        return descriptor

    def sift_vlfeat(self, img, res_name):
        cv2.imwrite('tmp.pgm', img)
        img_name = 'tmp.pgm'
        cmmd = str("sift " + img_name + " --output=" + res_name)
        os.system(cmmd)

    def feature_extraction(self, img):
        h, w = img.shape

        res_file_name = 'tmp.sift'

        self.sift_vlfeat(img, res_file_name)
        l, d = self.read_sift_from_vlfeat_file(res_file_name)

        descriptor_count = l.shape[0]
        descriptor_features = []
        img_features = {}

        for i in range(descriptor_count):
            descriptor = {}
            descriptor['x'] = l[i][0]
            descriptor['y'] = l[i][1]
            descriptor['vector'] = self.normalize_sift(d[i])
            descriptor_features.append(descriptor)

        img_features['file_name'] = 'tmp_img'
        img_features['height'] = h
        img_features['width'] = w
        img_features['descriptors'] = descriptor_features

        return img_features
