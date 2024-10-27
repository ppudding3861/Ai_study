#step1 : import modules


import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# assert insightface.__version__>='0.3'

# parser = argparse.ArgumentParser(description='insightface app test')
# # general
# parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
# parser.add_argument('--det-size', default=640, type=int, help='detection size')
# # args = parser.parse_args()


#step 2 : create inference object(instance)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))


# step 3 : load data
img = ins_get_image('t1')

# step 4 : inference
faces = app.get(img)
assert len(faces)==6

# step 5 : Post processing (application)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
feats = []
for face in faces:
    feats.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats, feats.T)
print(sims)