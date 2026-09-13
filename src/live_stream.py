"""Run MediaPipe Holistic landmarking on frames from a webcam."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "holistic_landmarker.task"
DEFAULT_CAT_VIDEO_PATH = Path(__file__).resolve().parents[1] / "assets" / "spinning_cat.mp4"
DEFAULT_CAT_MUSIC_PATH = Path(__file__).resolve().parents[1] / "assets" / "spinning_cat.wav"

POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
FACE_CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16


class HandsUpTrigger:
    """Debounce the hands-up pose so brief landmark noise does not flicker."""

    def __init__(self, hold_seconds: float = 0.35, release_seconds: float = 0.5):
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self._up_since: float | None = None
        self._down_since: float | None = None
        self.active = False

    def update(self, hands_are_up: bool, now: float) -> bool:
        if hands_are_up:
            self._down_since = None
            if self._up_since is None:
                self._up_since = now
            if now - self._up_since >= self.hold_seconds:
                self.active = True
        else:
            self._up_since = None
            if self._down_since is None:
                self._down_since = now
            if now - self._down_since >= self.release_seconds:
                self.active = False
        return self.active


class CatPlayer:
    """Loop a video in an OpenCV window and a WAV soundtrack on Windows."""

    WINDOW_NAME = "Spinning Cat"

    def __init__(self, video_path: Path, music_path: Path):
        if not video_path.is_file():
            raise FileNotFoundError(f"Cat video not found: {video_path}")
        if not music_path.is_file():
            raise FileNotFoundError(f"Cat music not found: {music_path}")

        try:
            import winsound
        except ImportError as error:
            raise RuntimeError("WAV playback currently requires Windows") from error

        self._winsound = winsound
        self._video = cv2.VideoCapture(str(video_path))
        if not self._video.isOpened():
            raise RuntimeError(f"Could not decode cat video: {video_path}")

        self._music_path = music_path
        self._frame_count = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._video.get(cv2.CAP_PROP_FPS) or 30.0
        self._duration = self._frame_count / self._fps if self._frame_count else 0.0
        self._started_at = 0.0
        self.active = False

    def start(self, now: float) -> None:
        if self.active:
            return
        self.active = True
        self._started_at = now
        self._winsound.PlaySound(
            str(self._music_path),
            self._winsound.SND_FILENAME
            | self._winsound.SND_ASYNC
            | self._winsound.SND_LOOP,
        )

    def update(self, now: float) -> None:
        if not self.active:
            return

        elapsed = now - self._started_at
        position = elapsed % self._duration if self._duration else elapsed
        self._video.set(cv2.CAP_PROP_POS_MSEC, position * 1000)
        ok, frame = self._video.read()
        if ok:
            cv2.imshow(self.WINDOW_NAME, frame)

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self._winsound.PlaySound(None, self._winsound.SND_PURGE)
        cv2.destroyWindow(self.WINDOW_NAME)

    def close(self) -> None:
        self.stop()
        self._video.release()


def _visible(landmark: object) -> bool:
    """Ignore pose points that MediaPipe says are absent or occluded."""
    visibility = getattr(landmark, "visibility", None)
    presence = getattr(landmark, "presence", None)
    return (visibility is None or visibility >= 0.5) and (
        presence is None or presence >= 0.5
    )


def _pixel(landmark: object, width: int, height: int) -> tuple[int, int]:
    x = max(0.0, min(1.0, float(getattr(landmark, "x"))))
    y = max(0.0, min(1.0, float(getattr(landmark, "y"))))
    return round(x * (width - 1)), round(y * (height - 1))


def draw_landmarks(
    frame: cv2.typing.MatLike,
    landmarks: Sequence[object],
    connections: Iterable[object],
    color: tuple[int, int, int],
    *,
    point_radius: int = 2,
) -> None:
    """Draw one MediaPipe landmark group directly onto an OpenCV frame."""
    if not landmarks:
        return

    height, width = frame.shape[:2]
    points = [_pixel(landmark, width, height) for landmark in landmarks]

    for connection in connections:
        start = int(getattr(connection, "start"))
        end = int(getattr(connection, "end"))
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        if _visible(landmarks[start]) and _visible(landmarks[end]):
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)

    if point_radius:
        for landmark, point in zip(landmarks, points, strict=True):
            if _visible(landmark):
                cv2.circle(frame, point, point_radius, color, -1, cv2.LINE_AA)


def annotate_frame(
    frame: cv2.typing.MatLike, result: vision.HolisticLandmarkerResult
) -> None:
    """Overlay face, pose, and hand landmarks in distinct colors."""
    draw_landmarks(
        frame, result.face_landmarks, FACE_CONNECTIONS, (180, 180, 180), point_radius=1
    )
    draw_landmarks(frame, result.pose_landmarks, POSE_CONNECTIONS, (0, 255, 0))
    draw_landmarks(frame, result.left_hand_landmarks, HAND_CONNECTIONS, (255, 80, 80))
    draw_landmarks(
        frame, result.right_hand_landmarks, HAND_CONNECTIONS, (80, 80, 255)
    )


def hands_up(pose_landmarks: Sequence[Any]) -> bool:
    """Return true when both visible wrists are above their shoulders."""
    if len(pose_landmarks) <= RIGHT_WRIST:
        return False

    required = (
        pose_landmarks[LEFT_SHOULDER],
        pose_landmarks[RIGHT_SHOULDER],
        pose_landmarks[LEFT_WRIST],
        pose_landmarks[RIGHT_WRIST],
    )
    if not all(_visible(landmark) for landmark in required):
        return False

    margin = 0.03
    return (
        pose_landmarks[LEFT_WRIST].y
        < pose_landmarks[LEFT_SHOULDER].y - margin
        and pose_landmarks[RIGHT_WRIST].y
        < pose_landmarks[RIGHT_SHOULDER].y - margin
    )


def run(
    camera: int,
    model: Path,
    cat_video: Path,
    cat_music: Path,
    width: int,
    height: int,
    mirror: bool,
) -> None:
    if not model.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {model}")

    cat_player = CatPlayer(cat_video, cat_music)
    capture = cv2.VideoCapture(camera)
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        capture.release()
        cat_player.close()
        raise RuntimeError(f"Could not open camera {camera}")

    hands_up_trigger = HandsUpTrigger()

    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    started_at = time.monotonic()
    last_timestamp_ms = -1

    try:
        with vision.HolisticLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")

                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((time.monotonic() - started_at) * 1000)
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                annotate_frame(frame, result)

                now = time.monotonic()
                cat_is_active = hands_up_trigger.update(
                    hands_up(result.pose_landmarks), now
                )
                if cat_is_active:
                    cat_player.start(now)
                    cat_player.update(now)
                else:
                    cat_player.stop()

                pose_status = "HANDS UP - CAT!" if cat_is_active else "Raise both hands"
                status_color = (0, 255, 255) if cat_is_active else (255, 255, 255)

                cv2.putText(
                    frame,
                    "Press Q or Esc to quit",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    pose_status,
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("MediaPipe Holistic", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cat_player.close()
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show MediaPipe Holistic landmarks from a webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--cat-video", type=Path, default=DEFAULT_CAT_VIDEO_PATH,
        help="spinning-cat MP4 file",
    )
    parser.add_argument(
        "--cat-music", type=Path, default=DEFAULT_CAT_MUSIC_PATH,
        help="WAV soundtrack played while the cat video is visible",
    )
    parser.add_argument("--width", type=int, default=1280, help="requested frame width")
    parser.add_argument("--height", type=int, default=720, help="requested frame height")
    parser.add_argument(
        "--no-mirror", action="store_true", help="do not mirror the webcam preview"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.camera,
        args.model.resolve(),
        args.cat_video.resolve(),
        args.cat_music.resolve(),
        args.width,
        args.height,
        not args.no_mirror,
    )


if __name__ == "__main__":
    main()
