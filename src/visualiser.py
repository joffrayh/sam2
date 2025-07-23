import torch
import cv2 as cv
import numpy as np

class Visualiser:
    '''
    - resizes mask and frame
    - overlays mask on frame
    '''
    def __init__(self, video_width, video_height):
        self.video_width = video_width
        self.video_height = video_height

    def resize_mask(self, mask):
        mask = mask.cpu()
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False)
        return mask

    def add_frame(self, frame, mask):
        frame = frame.copy()
        frame = cv.resize(frame, (self.video_width, self.video_height))
        
        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()

        alpha = 0.5
        pink = np.array([255, 105, 180])

        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            # add the alpha blended mask to the frame
            frame[obj_mask] = (
                pink * alpha + frame[obj_mask] * (1 - alpha)
            ).astype(np.uint8)
        
        return frame