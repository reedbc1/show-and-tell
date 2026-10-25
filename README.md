# Show-and-Tell

Run the gesture-controlled cat from the project directory:

```sh
uv run python src/main.py
```

Keep both shoulders and wrists visible to the webcam. Raise both wrists above
their corresponding shoulders to play the spinning cat GIF and loop the music.
Lower either wrist (or move out of view) to return to the stationary cat and stop
the music. Press `q` in the cat window or close it to exit.

The app uses the included `holistic_landmarker.task` model and files in `assets/`.
Wrist and shoulder positions come from [MediaPipe's pose landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

