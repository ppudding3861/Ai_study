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
# img = ins_get_image('t1')
img1 = cv2.imread('iu2.jpg')
img2 = cv2.imread('iu3.jpg')
# file open
# decode img



# step 4 : inference
faces1 = app.get(img1)
assert len(faces1)==1

faces2 = app.get(img2)
assert len(faces2)==1

# step 5 : Post processing (application)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity

emb1 = faces1[0].normed_embedding
emb2 = faces2[0].normed_embedding

# feats = []
# for face in faces:
#     feats.append(face.normed_embedding) # 위에 정의를 했기때문에 없어도 됨

np_emb1 = np.array(emb1, dtype=np.float32)
np_emb2 = np.array(emb2, dtype=np.float32)

sims = np.dot(np_emb1, np_emb2.T) # 행렬 연산해주는 np.dot
print(sims)