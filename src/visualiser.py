import torch
import cv2 as cv
import numpy as np
from matplotlib import colormaps as cm

class Visualiser:
    '''
    - resizes mask and frame
    - overlays mask on frame
    '''
    def __init__(self, video_width, video_height, alpha=0.5, colormap='viridis'):
        self.video_width = video_width
        self.video_height = video_height
        self.alpha = alpha
        self.colormap = cm.get_cmap(colormap)

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

        colors = self.colormap(np.linspace(0, 1, mask.shape[0]), bytes=True)[:, :3]

        for i, color in zip(range(mask.shape[0]), colors):
            obj_mask = mask[i, 0, :, :]
            # add the alpha blended mask to the frame
            frame[obj_mask] = (
                color * self.alpha + frame[obj_mask] * (1 - self.alpha)
            ).astype(np.uint8)
        
        return frame