#@markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

IMAGE_FILE = 'person2.jpg'

import cv2
# from google.colab.patches import cv2_imshow

# img = cv2.imread(IMAGE_FILE)
# # cv2_imshow(img)
# cv2.imshow("test", img)
# cv2.waitKey(0)

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                        score_threshold=0.3)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)
print(detection_result)

# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# 추론 결과에서 'person'을 감지한 수를 세는 함수
def count_person(detection_result):
    count = 0
    for detection in detection_result.detections:
        for category in detection.categories:
            if category.category_name == 'person':
                count += 1
    return count

# person의 수를 확인하고 출력
person_count = count_person(detection_result)
if person_count > 0:
    print(f"Person이 있습니다. 감지된 사람 수: {person_count}명")
else:
    print("Person이 없습니다.")

# 결과 이미지 표시
cv2.imshow("Detection Result", rgb_annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()