import time
import cv2
import numpy as np
import torch
import time
from sam2.build_sam import build_sam2_object_tracker

from visualiser import Visualiser


def collect_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['points'].append((x, y))
        print(f"Point added: ({x}, {y})")
        copy_frame = param['frame']
        cv2.circle(copy_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(param['window_name'], copy_frame)

NUM_OBJECTS = int(input("Enter number of objects to track: "))
VIDEO_STREAM = 0  # 0 is webcam
SAM_CHECKPOINT_FILEPATH = "./checkpoints/sam2.1_hiera_tiny.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_t.yaml"
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

# visualiser used to diplay object mask on the video
visualiser = Visualiser(video_width=video_width, video_height=video_height)

frame_number = 1
while True:

    print(f'Frame Number: {frame_number}\n')
    torch.cuda.synchronize()
    start = time.time()
    if frame_number == 1:

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        all_points = []
        for obj_num in range(NUM_OBJECTS):
            print(f"\nSelect points for object {obj_num + 1}:")
            print("\nLeft-click to select points. Press ENTER when done.")
            
            cv2.imshow("Select Object", frame)
            
            params={
                    'points': [],
                    'window_name': 'Select Object',
                    'frame': frame.copy(),
            }

            cv2.setMouseCallback(
                "Select Object", 
                collect_points, 
                param=params
            )

            key = 0
            while key != 13:
                key = cv2.waitKey(0)

                if cv2.getWindowProperty("Select Object", cv2.WND_PROP_VISIBLE) < 1:
                    exit()


            points_np = np.array(params['points'])
            print(f"Points selected for object {obj_num + 1}:\n", points_np)
            all_points.append(points_np)
        
        cv2.destroyWindow("Select Object")

        # stack all points into shape (NUM_OBJECTS, max_points, 2)
        max_points = max(len(p) for p in all_points)
        stacked_points = np.zeros((NUM_OBJECTS, max_points, 2), dtype=np.float32)
        for i, pts in enumerate(all_points):
            stacked_points[i, :len(pts)] = pts
        sam.curr_obj_idx = 0
        sam_out = sam.track_new_object(
            img=img_rgb,
            points=stacked_points,
        )

    else:
        ret, frame = video_stream.read()
        if not ret:
            raise RuntimeError("Failed to read from webcam.")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            sam_out = sam.track_all_objects(img=img_rgb)

    torch.cuda.synchronize()
    print(f'[TIMING] Frame {frame_number} processing took {1000*(time.time() - start):.1f} ms')
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

    # visualise and save frame
    processed_frame = visualiser.add_frame(frame=frame, mask=sam_out['pred_masks'])

    cv2.imshow("Frame", processed_frame)

    frame_number+=1
    torch.cuda.empty_cache()


    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

    print('\n-------------------------------------------------------------\n')

video_stream.release()
cv2.destroyAllWindows()