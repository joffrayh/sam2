import time
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_object_tracker

from visualiser import Visualiser


points = []

def collect_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

# configs
VIDEO_STREAM = './juggle.mp4'
NUM_OBJECTS = 1
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_tiny.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_t.yaml"
OUTPUT_PATH = VIDEO_STREAM + "_segmented.mp4"
SAVE = True
DEVICE = 'cuda:0'

video_stream = cv2.VideoCapture(VIDEO_STREAM)

video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

visualiser = Visualiser(video_width=video_width, video_height=video_height)

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False)

# video writer for output
if SAVE:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (video_width, video_height))

frame_number = 1
while video_stream.isOpened():
    ret, frame = video_stream.read()
    if not ret:
        break

    # get user input to select points on the first frame
    print(f'Frame Number: {frame_number}\n')
    if frame_number == 1:

        cv2.imshow("Select Object", frame)
        cv2.setMouseCallback("Select Object", collect_points)

        print("Click to select points on the object. Press ENTER when done.")

        while True:
            # show selected points
            frame_copy = frame.copy()
            for pt in points:
                cv2.circle(frame_copy, pt, radius=5, color=(0, 255, 0), thickness=-1)
            cv2.imshow("Select Object", frame_copy)

            key = cv2.waitKey(1)

            if key == 13:  # 'enter' key
                break

            # check if the window has been closed
            if cv2.getWindowProperty("Select Object", cv2.WND_PROP_VISIBLE) < 1:
                video_stream.release()
                if SAVE:
                    out.release()
                cv2.destroyAllWindows()
                exit(0)

        cv2.destroyWindow("Select Object")

        if len(points) == 0:
            print("No points selected. Exiting.")
            break

        points_np = np.array([points])
        print(f"Selected points:\n{points_np}")
        print(points_np.shape)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        sam_out = sam.track_new_object(img=img_rgb, points=points_np)

    else:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            sam_out = sam.track_all_objects(img=img_rgb)


    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

    # Visualise and save frame
    start = time.time()
    processed_frame = visualiser.add_frame(frame=frame, mask=sam_out['pred_masks'])

    if SAVE:
        out.write(processed_frame)

    cv2.imshow("Frame", processed_frame)

    frame_number+=1
    torch.cuda.empty_cache()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print('\n-------------------------------------------------------------\n')

# save
if SAVE:
    out.release()
cv2.destroyAllWindows()