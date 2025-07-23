import time
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_object_tracker

from visualiser import Visualiser

drawing = False
ix, iy = -1, -1
bbox = []
points = []

mode_selected = False
use_points = False

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Object", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = [(ix, iy), (x, y)]
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Select Object", frame)

def collect_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

# configs
VIDEO_STREAM = 0  # 0 is webcam
NUM_OBJECTS = 1
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_tiny.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_t.yaml"
OUTPUT_PATH = "webcam_output.mp4"
SAVE = False
DEVICE = 'cuda:0'

video_stream = cv2.VideoCapture(VIDEO_STREAM)

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False)

# reading the first frame to get dimensions
ret, frame = video_stream.read()
if not ret:
    raise RuntimeError("Failed to read from webcam.")
video_height, video_width = frame.shape[:2]

# video writer for output
if SAVE:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (video_width, video_height))

# visualiser used to diplay object mask on the video
visualiser = Visualiser(video_width=video_width, video_height=video_height)

frame_number = 1
while True:

    # get user input to select bounding box
    print(f'Frame Number: {frame_number}\n')

    if frame_number == 1:

        use_points = input("Use points (p) or bounding box (b)? ").strip().lower() == 'p'
        mode_selected = True

        ret, frame = video_stream.read()
        if not ret:
            raise RuntimeError("Failed to read from webcam.")

        cv2.imshow("Select Object", frame)

        if use_points:
            cv2.setMouseCallback("Select Object", collect_points)
            print("Left-click to select points. Press ENTER when done.")

            while True:
                frame_copy = frame.copy()
                for pt in points:
                    cv2.circle(frame_copy, pt, 5, (0, 255, 0), -1)
                cv2.imshow("Select Object", frame_copy)
                key = cv2.waitKey(1)
                if key == 13:  # Enter
                    break
                if cv2.getWindowProperty("Select Object", cv2.WND_PROP_VISIBLE) < 1:
                    exit()
            cv2.destroyWindow("Select Object")
            points_np = np.array([points])
            print("Points selected:", points_np)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sam_out = sam.track_new_object(img=img_rgb, points=points_np)

        else:
            cv2.setMouseCallback("Select Object", draw_rectangle)
            print("Draw bounding box by left-clicking and dragging.")
            while len(bbox) == 0:
                cv2.waitKey(1)
            cv2.destroyWindow("Select Object")
            bbox_np = np.array([bbox])
            print("BBox selected:", bbox_np)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sam_out = sam.track_new_object(img=img_rgb, box=bbox_np)

    else:
        ret, frame = video_stream.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            
            sam_out = sam.track_all_objects(img=img_rgb)


    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

    # visualise and save frame
    start = time.time()
    processed_frame = visualiser.add_frame(frame=frame, mask=sam_out['pred_masks'])

    if SAVE:
        out.write(processed_frame)

    cv2.imshow("Frame", processed_frame)

    frame_number+=1
    torch.cuda.empty_cache()


    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

    print('\n-------------------------------------------------------------\n')

# save
if SAVE:
    out.release()
cv2.destroyAllWindows()