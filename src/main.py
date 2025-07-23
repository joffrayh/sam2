import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_object_tracker


class Visualizer:
    def __init__(self, video_width, video_height):
        self.video_width = video_width
        self.video_height = video_height

    def resize_mask(self, mask):
        # mask = torch.tensor(mask, device='cpu')
        mask = mask.cpu()
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False)
        return mask

    def add_frame(self, frame, mask):
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()

        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            frame[obj_mask] = [255, 105, 180]  # Color the object region
        
        return frame  # Return the modified frame to show/save later


drawing = False
ix, iy = -1, -1
bbox = []

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

# configs
VIDEO_STREAM = './juggle.mp4'
NUM_OBJECTS = 1
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
SAM_CHECKPOINT_FILEPATH = "../sam2_realtime/checkpoints/sam2.1_hiera_tiny.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_t.yaml"
OUTPUT_PATH = VIDEO_STREAM + "_segmented.mp4"
DEVICE = 'cuda:0'

video_stream = cv2.VideoCapture(VIDEO_STREAM)

video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

visualizer = Visualizer(video_width=video_width, video_height=video_height)

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False)

# video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (video_width, video_height))

frame_number = 1
while video_stream.isOpened():
    ret, frame = video_stream.read()
    if not ret:
        break

    # get user input to select bounding box
    print(f'Frame Number: {frame_number}\n')
    if frame_number == 1:

        cv2.imshow("Select Object", frame)
        cv2.setMouseCallback("Select Object", draw_rectangle)

        # wait for the box to be selected
        while len(bbox) == 0:
            cv2.waitKey(1)

        cv2.destroyWindow("Select Object")
        

        bbox = np.array([bbox])
        print(f"selected bbox: \n{bbox}")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        sam_out = sam.track_new_object(img=img_rgb, box=bbox)
        
        first_frame = False
    else:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            sam_out = sam.track_all_objects(img=img_rgb)


    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
    print('\n-------------------------------------------------------------\n')

    # visualize and save frame
    processed_frame = visualizer.add_frame(frame=frame, mask=sam_out['pred_masks'])
    out.write(processed_frame)

    cv2.imshow("Frame", processed_frame)

    frame_number+=1

    torch.cuda.empty_cache()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# save
out.release()
cv2.destroyAllWindows()