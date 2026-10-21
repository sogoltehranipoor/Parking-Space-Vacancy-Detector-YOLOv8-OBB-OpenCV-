# Parking-Space-Vacancy-Detector-YOLOv8-OBB-OpenCV-
A computer vision project that watches a top-down video of a parking lot, checks pre-defined parking spots, and colors each one **green (available)** or **red (occupied)**, with a live counter such as `Available: 11 / 13`.

## How it works

1. **Define the spots.** The parking row is described by a few fractions of the frame size (left/right edge, top/bottom of the white lines, number of spots), so the layout does not depend on video resolution.
2. **Detect vehicles.** A pretrained **YOLOv8s-OBB** model (trained on the DOTA aerial dataset, classes `small vehicle` and `large vehicle`) runs on each frame at three input sizes (320, 480, 1024). The detections are merged with non-maximum suppression.
3. **Decide occupancy.** For every spot, the fraction of its area covered by a detected vehicle box is computed. If it is above a threshold (default 30%), the spot is occupied.
4. **Smooth the result.** A majority vote over the last 7 frames per spot (using `collections.deque`) removes flicker when the detector misses a car for a frame or two.
5. **Draw the output.** Green/red boxes and the availability counter are drawn on the video frames.

## Why YOLOv8-OBB instead of the standard YOLO model?

The standard COCO-trained YOLOv8 models detected **no cars at all** in the top-down footage, even at very low confidence thresholds. Switching to an oriented-bounding-box model trained on aerial imagery fixed this. Results also depended strongly on the input size (640 found nothing, 320/480/1024 each found a car), so the final pipeline merges several sizes.

## Tech stack

- Python, Google Colab (T4 GPU)
- [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8-OBB
- OpenCV, NumPy, Matplotlib

## Run it

1. Open the notebook in Google Colab and set the runtime to a GPU.
2. Run the cells in order: install and imports, upload your video, adjust the spot layout until the magenta boxes sit on the parking spots, check the detections on the first frame, then run on the full video.
3. The last cell shows the processed frames inside the notebook.

## Main settings

| Setting | Default | Meaning |
|---|---|---|
| `SCALES` | `[320, 480, 1024]` | YOLO input sizes that are run and merged |
| `CONF` | `0.05` | Detection confidence (low, duplicates are merged afterwards) |
| `NMS_IOU` | `0.4` | Overlap above which detections are merged |
| `OVERLAP_THR` | `0.30` | Fraction of a spot a vehicle must cover to mark it occupied |
| `SMOOTH_FRAMES` | `7` | Frames used for the majority vote |

## Limitations

- Tested on a single fixed-camera, top-down video; no quantitative evaluation (precision/recall) yet.
- One row of spots is defined by hand; other rows would need their own layout.
- Thresholds were tuned by eye.
- Cars that are only partly visible at the frame edge may be missed.

## Ideas for future work

- Click-based or automatic spot detection
- Fine-tuning YOLO on a parking dataset (for example PKLot)
- Evaluation on labeled frames
- Multiple rows and perspective views
- Exporting the result as an MP4

## Notes

The demo video is not included in this repository. Add your own top-down parking video and check its license before publishing it.
