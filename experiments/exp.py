from pathlib import Path
import sys
import time

import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PROJECT_DIR = Path(__file__).resolve().parent.parent
model_path = PROJECT_DIR / "holistic_landmarker.task"


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks = detection_result.pose_landmarks
  face_landmarks = detection_result.face_landmarks
  left_hand_landmarks = detection_result.left_hand_landmarks
  right_hand_landmarks = detection_result.right_hand_landmarks
  annotated_image = np.copy(rgb_image[:, :, :3] if rgb_image.shape[2] == 4 else rgb_image)

  # Draw pose landmarks.
  if pose_landmarks:
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  # Draw face landmarks.
  if face_landmarks:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  # Draw hand landmarks.
  if left_hand_landmarks:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=left_hand_landmarks,
        connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=drawing_styles.get_default_hand_connections_style())

  if right_hand_landmarks:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=right_hand_landmarks,
        connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=drawing_styles.get_default_hand_connections_style())

  return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


def main():
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5)

    with HolisticLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        cap = cv.VideoCapture(0)
        start = time.perf_counter()
        if not cap.isOpened():
            print("Cannot open camera.")
            sys.exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't recieve frame (stream end?). Exiting...")
                break
            # Our operations on the frame come here
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            time_ms = int((time.perf_counter() - start) * 1000)

            result = landmarker.detect_for_video(mp_image, time_ms)
            get_shoulders(result)

            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)

            # Display the resulting frame
            cv.imshow("frame", annotated_image)

            if cv.waitKey(1) == ord("q"):
                break

        ## When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()


def get_shoulders(result):
    ls = result.pose_landmarks[vision.PoseLandmark.LEFT_SHOULDER]
    rs = result.pose_landmarks[vision.PoseLandmark.RIGHT_SHOULDER]
    print(f"left shoulder: {ls}")
    print(f"right shoulder: {rs}")


if __name__ == "__main__":
   main()