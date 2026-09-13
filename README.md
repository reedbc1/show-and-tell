# Webcam holistic landmarking

This example reads frames from a webcam with OpenCV and processes each frame
with the Google MediaPipe Holistic Landmarker. It displays face contours, pose
landmarks, and both hands in a live preview. Holding both hands above the
shoulders opens a looping spinning-cat video with music.

## Run

Create an `assets` directory and add these two media files:

- `assets/spinning_cat.mp4` — the video
- `assets/spinning_cat.wav` — its music or soundtrack as a WAV file

OpenCV does not play an MP4's embedded audio, so the soundtrack is a separate
WAV file. Install the locked dependencies and start the camera:

```powershell
uv sync
uv run python -m src.live_stream
```

Press `Q` or `Esc` to close the preview. The default webcam is device `0`; use a
different device or resolution when needed:

```powershell
uv run python -m src.live_stream --camera 1 --width 640 --height 480
```

Custom media paths can also be supplied:

```powershell
uv run python -m src.live_stream --cat-video C:\media\cat.mp4 --cat-music C:\media\cat.wav
```

Run `uv run python -m src.live_stream --help` for all options. The model path is
resolved from the project directory, so the command also works when launched
from another current working directory.
