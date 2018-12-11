import cv2
import json
import numpy as np

from flask import Flask, request, render_template

from logic.sift_feature_extraction import FeatureExtraction
from logic.spatial_pyramid_histogram import SpatialPyramidHistogram
from logic.image_retrieval import ImageRetrieval

app = Flask(__name__)


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

    return render_template('index.html', similar_images=similar_img)

if __name__ == '__main__':
    app.run()
