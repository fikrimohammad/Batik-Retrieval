import numpy as np
import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity


class ImageRetrieval(object):

    def read_models_from_file(self, filename):
        file = open(filename, "rb")
        models = pickle.load(file)
        file.close()
        return models

    def retrieve_similar_img(self, img_feature, training_features):
        cos_similarity = []
        for feature in training_features:
            value = cosine_similarity(img_feature['histogram'].reshape(1, -1),
                                      feature['histogram'].reshape(1, -1))
            cos_similarity.append(value.flatten())

        #    sorted_idx = [index for index, value in sorted(enumerate(cos_similarity),
        #                                                   reverse=True,
        #                                                   key=lambda x: x[1])]

        sorted_training_features = [list(x) for x in zip(*sorted(zip(cos_similarity, training_features),
                                                                 key=lambda pair: pair[0],
                                                                 reverse=True)[0:30])]

        return sorted_training_features[1]

    def top_ten_image_retrieval(self, img_feature):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        training_img_path = os.path.join(BASE_DIR, '../machine-learning-models/all_img_histogram.sav')

        training_img_features = self.read_models_from_file(training_img_path)
        similar_img = self.retrieve_similar_img(img_feature, training_img_features)

        similar_img_name = []
        for img in similar_img:
            similar_img_name.append(img['file_name'])

        return similar_img_name