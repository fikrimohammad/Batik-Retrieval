import cv2
import json
import numpy as np

from flask import Flask, request, render_template, url_for

from logic.sift_feature_extraction import FeatureExtraction
from logic.spatial_pyramid_histogram import SpatialPyramidHistogram
from logic.image_retrieval import ImageRetrieval

app = Flask(__name__)
# app._static_folder = "D:\Tugas Kuliah\Visi Komputer\FP-Web\static"


@app.route('/', methods=['GET'])
def search_form():
    return render_template('search_form.html')


@app.route('/', methods=['POST'])
def batik_retrieval():

    fe = FeatureExtraction()
    sph = SpatialPyramidHistogram()
    ir = ImageRetrieval()

    filestr = request.files['img'].read()

    npimg = np.fromstring(filestr, np.uint8)
    batik_img = cv2.imdecode(npimg, 0)

    img_sift_features = fe.feature_extraction(batik_img)

    img_histogram = sph.build_histogram(img_sift_features)

    similar_img = ir.top_ten_image_retrieval(img_histogram)
    nama = 'Batik '+ similar_img[0][9]
    if (similar_img[0][10] != '_'):
        nama = nama + similar_img[0][10]

    return render_template('search_result.html', similar_images=similar_img, namabatik=nama)


if __name__ == '__main__':
    app.run()
