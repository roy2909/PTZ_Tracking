## Boat Tracking Application 

This is a Python based real time tracking system that uses YOLOv8 for object detection and PID controllers for stabilizing pan, tilt, and digital zoom. It allows users to track boats in a video feed (either from a webcam or file) and zoom in on the target object smoothly. Users can manually select the boat they want to track if multiple are detected.



[boats_final.webm](https://github.com/user-attachments/assets/1018a736-dcd6-482b-8645-3aab049e21fc)



### Requirements

- Python 3.8 or higher
- OpenCV
- Ultralytics YOLOv8
- NumPy

### Installation

1. Clone the repository 
   `git clone git@github.com:roy2909/PTZ_Tracking.git`

2. Install dependencies
   `pip install opencv-python numpy torch ultralytics`

3. Download YOLOv8 model
   `The script will automatically download yolov8n.pt if not present
 Or you can manually download from https://github.com/ultralytics/ultralytics`

### Usage

1. python3 ptz_boat_tracker.py [-h] (--video VIDEO | --webcam) 
                       [--output OUTPUT] 
                       [--model MODEL] 
                       [--confidence CONFIDENCE]

| Argument                  | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `-h, --help`              | Show help message and exit                       |
| `--video VIDEO`           | Path to input video file                         |
| `--webcam`                | Use webcam as input source                       |
| `--output OUTPUT`         | Path to save output video (only for video input) |
| `--model MODEL`           | Path to YOLOv8 model (`.pt` file)                |
| `--confidence CONFIDENCE` | Detection confidence threshold (e.g., `0.5`)     |

### Examples
1. To track boats in a video file:
   `
   python3 ptz_boat_tracker.py --video path_to_video.mp4 --output output_video.mp4
   `

2. To track boats using a webcam:
   `
    python3 ptz_boat_tracker.py --webcam
    `


