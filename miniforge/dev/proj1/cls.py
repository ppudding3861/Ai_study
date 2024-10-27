# 가상환경 목록 조회
# conda env list

# 가상환경 만들기
# conda create -n project1 python=3.12

# 가상환경 진입하기
# conda activate project1

# 가상환경 나가기
# conda deactivate

# 가상환경 삭제하기
# conda remove -n project1

# Vectorizing 패키지
# pip install numpy

# import urllib.request

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

import cv2
# from google.colab.patches import cv2_imshow
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
#   cv2_imshow(img)
  cv2.imshow("test", img)
  cv2.waitKey(0)


# Preview the images.
# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)



# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file('burger.jpg')

# STEP 4: Classify the input image.
classification_result = classifier.classify(image)
# print(classification_result)

# ClassificationResult(
#     classifications=[
#         Classifications(
#             categories=[
#                 Category(index=933, score=0.9790241718292236, display_name='', category_name='cheeseburger'), 
#                 Category(index=931, score=0.0008637138525955379, display_name='', category_name='bagel'), 
#                 Category(index=947, score=0.0005722686764784157, display_name='', category_name='mushroom')
#                 ], 
#                 head_index=0, head_name='probability')
#     ], 
#     timestamp_ms=0
#     )

# STEP 5: Process the classification result. In this case, visualize it.

top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")
