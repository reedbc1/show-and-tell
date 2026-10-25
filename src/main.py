import time
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PROJECT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_DIR / "assets"
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

  fig, ax = plt.subplots(figsize=(12, 12))  # noqa: RUF059
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

    cap = None
    cat_cap = None
    try:
        stationary_cat = cv.imread(str(ASSETS_DIR / "stationary-cat.png"))
        if stationary_cat is None:
            raise RuntimeError("Cannot load assets/stationary-cat.png.")

        cat_cap = cv.VideoCapture(str(ASSETS_DIR / "spinning-cat.gif"))
        if not cat_cap.isOpened():
            raise RuntimeError("Cannot open assets/spinning-cat.gif.")

        pygame.mixer.init()
        pygame.mixer.music.load(str(ASSETS_DIR / "ooeeii.ogg"))

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera.")

        with HolisticLandmarker.create_from_options(options) as landmarker:
            spinning = False
            start = time.perf_counter()
            last_timestamp_ms = -1

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Cannot receive a camera frame. Exiting...")
                    break

                # OpenCV captures BGR; MediaPipe expects RGB.
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                time_ms = max(last_timestamp_ms + 1, int((time.perf_counter() - start) * 1000))
                last_timestamp_ms = time_ms
                result = landmarker.detect_for_video(mp_image, time_ms)
                should_spin = wrists_over_shoulders(result)

                # Change playback only when the gesture changes, so music can loop.
                if should_spin != spinning:
                    spinning = should_spin
                    if spinning:
                        cat_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                        pygame.mixer.music.play(loops=-1, start=0.25)
                    else:
                        pygame.mixer.music.stop()

                cat_frame = stationary_cat
                if spinning:
                    cat_ret, cat_frame = cat_cap.read()
                    if not cat_ret:
                        # Rewind the GIF, not the webcam, when the animation ends.
                        cat_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                        cat_ret, cat_frame = cat_cap.read()
                    if not cat_ret:
                        raise RuntimeError("Cannot read a frame from assets/spinning-cat.gif.")

                cv.imshow("GIF Player", cat_frame)
                # Keep checking the camera even while the stationary image is shown.
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv.getWindowProperty("GIF Player", cv.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if cap is not None:
            cap.release()
        if cat_cap is not None:
            cat_cap.release()
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        cv.destroyAllWindows()


def wrists_over_shoulders(result, min_visibility=0.5):
    """Return whether both visible wrists are above their matching shoulders."""
    landmarks = result.pose_landmarks
    if not landmarks or len(landmarks) <= vision.PoseLandmark.RIGHT_WRIST:
        return False

    ls = landmarks[vision.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[vision.PoseLandmark.RIGHT_SHOULDER]
    lw = landmarks[vision.PoseLandmark.LEFT_WRIST]
    rw = landmarks[vision.PoseLandmark.RIGHT_WRIST]
    if any(point.visibility is None or point.visibility < min_visibility for point in (ls, rs, lw, rw)):
        return False

    # Image y coordinates increase downward, so a smaller y is higher.
    return lw.y < ls.y and rw.y < rs.y


if __name__ == "__main__":
    main()
