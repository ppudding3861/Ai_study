import cv2
from mediapipe import solutions
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1

def draw_landmarks_on_image(rgb_image, detection_result):
    # Check and convert the image to 3 channels if needed
    if len(rgb_image.shape) == 2:  # Grayscale image
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    elif rgb_image.shape[2] == 4:  # RGBA image
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)

    annotated_image = np.copy(rgb_image)

    # Iterate over each detected pose landmark in the list
    if detection_result.pose_landmarks:  # Ensure there are landmarks
        for pose_landmarks in detection_result.pose_landmarks:
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks,  # Access each NormalizedLandmarkList
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create a PoseLandmarker object.
try:
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
except RuntimeError as e:
    print(f"Error initializing PoseLandmarker: {e}")
    exit(1)

# STEP 3: Load the input image.
try:
    image = mp.Image.create_from_file("girl-4051811_960_720.jpg")
except FileNotFoundError as e:
    print(f"Error loading image: {e}")
    exit(1)

# STEP 4: Detect pose landmarks from the input image.
try:
    detection_result = detector.detect(image)
except Exception as e:
    print(f"Error detecting pose landmarks: {e}")
    exit(1)

# STEP 5: Convert the image to numpy array and ensure it is in RGB format
image_np = image.numpy_view()

# 이미지가 올바른 형식인지 확인하고 필요한 경우 변환합니다.
if len(image_np.shape) == 2:  # Grayscale인 경우 RGB로 변환
    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
elif image_np.shape[2] == 4:  # RGBA인 경우 RGB로 변환
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

# STEP 6: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image_np, detection_result)

# Display the image.
cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
