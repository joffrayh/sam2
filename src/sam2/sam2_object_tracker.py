import time
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sam2.modeling.sam2_base import SAM2Base, NO_OBJ_SCORE
from sam2.modeling.sam2_utils import get_1d_sine_pe
from sam2.utils.kalman_filter import KalmanFilter

from tabulate import tabulate


class SAM2ObjectTracker(SAM2Base):
    def __init__(self,
                 num_objects=10,
                 verbose=True,
                 samurai_mode=True,
                 stable_frames_threshold: int = 15,
                 stable_ious_threshold: float = 0.3,
                 min_obj_score_logits: float = -1,
                 kf_score_weight: float = 0.15,
                 memory_bank_iou_threshold: float = 0.5,
                 memory_bank_obj_score_threshold: float = 0.0,
                 memory_bank_kf_score_threshold: float = 0.0,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self.num_objects = num_objects
        self.curr_obj_idx = 0

        self.model_constants = {}

        self.past_frames = {'short_term': deque(maxlen=7), 'long_term': deque(maxlen=7)}
        self.verbose = verbose
        self.use_mask_input_as_output_without_sam = False

        # Init Kalman Filter
        self.kf = KalmanFilter()
        self.kf_mean = {}
        self.kf_covariance = {}
        self.stable_frames = {}

        # Hyperparameters for SAMURAI
        self.stable_frames_threshold = stable_frames_threshold
        self.stable_ious_threshold = stable_ious_threshold
        self.min_obj_score_logits = min_obj_score_logits
        self.kf_score_weight = kf_score_weight
        self.memory_bank_iou_threshold = memory_bank_iou_threshold
        self.memory_bank_obj_score_threshold = memory_bank_obj_score_threshold
        self.memory_bank_kf_score_threshold = memory_bank_kf_score_threshold

    def update_kalman_filter(self,
                             obj: int,
                             ious: torch.Tensor,
                             low_res_multimasks: torch.Tensor,
                             high_res_multimasks: torch.Tensor,
                             sam_output_tokens: torch.Tensor
                             ) -> Tuple:
        """
        Updates the Kalman filter for object tracking based on the current object, IoU scores,
        low and high resolution multi-masks, and SAM output tokens.

        The original code can be found in the SAMURAI repo and has been adapted to work with multiple objects:
        https://github.com/yangchris11/samurai/blob/master/sam2/sam2/modeling/sam2_base.py#L421-L509

        Parameters
        ----------
        obj : int
            The identifier for the object being tracked.

        ious : torch.Tensor
            A tensor containing the intersection over union (IoU) values for the current object
            across multiple possible detections.

        low_res_multimasks : torch.Tensor
            A tensor containing low resolution multi-masks for the object detections.

        high_res_multimasks : torch.Tensor
            A tensor containing high resolution multi-masks for the object detections.

        sam_output_tokens : torch.Tensor
            A tensor containing the output tokens from the SAM model for each detection.

        Returns
        -------
        Tuple
            A tuple containing the following:
            - low_res_masks : torch.Tensor
                The low resolution mask corresponding to the best IoU detection.

            - high_res_masks : torch.Tensor
                The high resolution mask corresponding to the best IoU detection.

            - sam_output_token : torch.Tensor
                The SAM output token corresponding to the best IoU detection.

            - ious : torch.Tensor
                The IoU score of the best detection after filtering.

            - kf_ious : torch.Tensor
                The Kalman filter IoU score of the best detection after filtering.

        """

        ious = ious[obj]
        low_res_multimasks = low_res_multimasks[obj]
        high_res_multimasks = high_res_multimasks[obj]
        sam_output_tokens = sam_output_tokens[obj]

        kf_ious = torch.full((1,), float('nan'),  device=self.device)

        if obj not in self.kf_mean or self.stable_frames.get(obj, 0) == 0:
            best_iou_inds = torch.argmax(ious, dim=-1)

            non_zero_indices = torch.argwhere(high_res_multimasks[best_iou_inds] > 0.0)

            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]

            else:
                y_min, x_min = non_zero_indices.min(dim=0).values
                y_max, x_max = non_zero_indices.max(dim=0).values
                high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

            kf_mean, kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))
            self.kf_mean[obj] = kf_mean
            self.kf_covariance[obj] = kf_covariance

            self.stable_frames[obj] = 1

        elif self.stable_frames[obj] < self.stable_frames_threshold:
            kf_mean, kf_covariance = self.kf.predict(self.kf_mean[obj], self.kf_covariance[obj])
            self.kf_mean[obj] = kf_mean
            self.kf_covariance[obj] = kf_covariance

            best_iou_inds = torch.argmax(ious, dim=-1)

            non_zero_indices = torch.argwhere(high_res_multimasks[best_iou_inds] > 0.0)

            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]

            else:
                y_min, x_min = non_zero_indices.min(dim=0).values
                y_max, x_max = non_zero_indices.max(dim=0).values
                high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

            if ious[best_iou_inds] > self.stable_ious_threshold:
                kf_mean, kf_covariance = self.kf.update(self.kf_mean[obj],
                                                        self.kf_covariance[obj],
                                                        self.kf.xyxy_to_xyah(high_res_bbox)
                                                        )
                self.kf_mean[obj] = kf_mean
                self.kf_covariance[obj] = kf_covariance

                self.stable_frames[obj] += 1

            else:
                self.stable_frames[obj] = 0

        else:
            kf_mean, kf_covariance = self.kf.predict(self.kf_mean[obj], self.kf_covariance[obj])

            self.kf_mean[obj] = kf_mean
            self.kf_covariance[obj] = kf_covariance

            high_res_multibboxes = []

            for i in range(0, len(ious)):
                non_zero_indices = torch.argwhere(high_res_multimasks[i] > 0.0)

                if len(non_zero_indices) == 0:
                    high_res_multibboxes.append([0, 0, 0, 0])

                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_multibboxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])

            # compute the IoU between the predicted bbox and the high_res_multibboxes
            kf_ious = torch.tensor(self.kf.compute_iou(self.kf_mean[obj][:4], high_res_multibboxes), device=self.device)

            # weighted iou
            weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
            best_iou_inds = torch.argmax(weighted_ious, dim=-1)

            if ious[best_iou_inds] < self.stable_ious_threshold:
                self.stable_frames[obj] = 0

            else:
                kf_mean, kf_covariance = self.kf.update(self.kf_mean[obj],
                                                        self.kf_covariance[obj],
                                                        self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds])
                                                        )
                self.kf_mean[obj] = kf_mean
                self.kf_covariance[obj] = kf_covariance

        low_res_masks = low_res_multimasks[best_iou_inds].unsqueeze(0)
        high_res_masks = high_res_multimasks[best_iou_inds].unsqueeze(0)
        sam_output_token = sam_output_tokens[best_iou_inds].unsqueeze(0)
        ious = ious[best_iou_inds].unsqueeze(0)
        kf_ious = kf_ious[best_iou_inds].unsqueeze(0) if ~torch.isnan(kf_ious).all() else kf_ious

        return low_res_masks, high_res_masks, sam_output_token, ious, kf_ious

    def forward_sam_heads(self,
                          backbone_features,
                          point_inputs=None,
                          mask_inputs=None,
                          high_res_features=None,
                          multimask_output=False,
                          ) -> Tuple:
        """
        Forward SAM prompt encoders and mask heads.

        Parameters
        ----------
        backbone_features : torch.Tensor
            Image features of shape [B, C, H, W], where B is batch size,
            C is the number of channels, and H and W are height and width.

        point_inputs : dict, optional
            A dictionary containing "point_coords" and "point_labels":
            - "point_coords" has shape [B, P, 2] and dtype float32, containing
              the absolute pixel coordinates (x, y) of the P input points.
            - "point_labels" has shape [B, P] and dtype int32, where 1 means
              positive clicks, 0 means negative clicks, and -1 means padding.

        mask_inputs : torch.Tensor, optional
            A mask tensor of shape [B, 1, H*16, W*16], with float or bool values.
            This tensor has the same spatial size as the image and is used for masking.

        high_res_features : Union[None, list], optional
            Either None or a list of length 2 containing two feature maps:
            - The first has shape [B, C, 4*H, 4*W].
            - The second has shape [B, C, 2*H, 2*W].
            These features will be used as high-resolution feature maps for the SAM decoder.

        multimask_output : bool, optional
            If True, the output includes 3 candidate masks and their corresponding IoU estimates.
            If False, only 1 mask and its corresponding IoU estimate are returned.

        Returns
        -------
        Tuple
            A tuple containing the following elements:
            - low_res_multimasks : torch.Tensor
                A tensor of shape [B, M, H*4, W*4], where M = 3 if `multimask_output=True`
                and M = 1 if `multimask_output=False`, representing the low-resolution
                mask logits (before sigmoid) for the masks, with 4x the resolution of the
                input backbone_features.

            - high_res_multimasks : torch.Tensor
                A tensor of shape [B, M, H*16, W*16], where M = 3 if `multimask_output=True`
                and M = 1 if `multimask_output=False`, upsampled from the low-resolution masks
                to the size of the image (stride of 1 pixel).

            - ious : torch.Tensor
                A tensor of shape [B, M], containing the estimated IoU for each output mask.

            - low_res_masks : torch.Tensor
                A tensor of shape [B, 1, H*4, W*4], representing the best mask from
                `low_res_multimasks`. If `multimask_output=True`, this is the mask with
                the highest IoU estimate. If `multimask_output=False`, it's the same as
                `low_res_multimasks`.

            - high_res_masks : torch.Tensor
                A tensor of shape [B, 1, H*16, W*16], representing the best mask from
                `high_res_multimasks`. If `multimask_output=True`, this is the mask with
                the highest IoU estimate. If `multimask_output=False`, it's the same as
                `high_res_multimasks`.

            - obj_ptr : torch.Tensor
                A tensor of shape [B, C], representing the object pointer vector for
                the output mask, extracted based on the output token from the SAM mask decoder.

        """

        B = backbone_features.size(0)
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B

        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(self.num_objects, 1, 2, device=self.device)
            sam_point_labels = -torch.ones(self.num_objects, 1, dtype=torch.int32, device=self.device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(mask_inputs.float(),
                                                size=self.sam_prompt_encoder.mask_input_size,
                                                align_corners=False,
                                                mode="bilinear",
                                                antialias=True,  # use antialias for downsampling
                                                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        if point_inputs is not None or mask_inputs is not None:
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points=(sam_point_coords, sam_point_labels),
                                                                          boxes=None,
                                                                          masks=sam_mask_prompt,
                                                                          )

        else:
            if 'sparse_embeddings' not in self.model_constants:
                sam_point_coords = torch.zeros(self.num_objects, 1, 2, device=self.device)
                sam_point_labels = -torch.ones(self.num_objects, 1, dtype=torch.int32, device=self.device)

                sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points=(sam_point_coords, sam_point_labels),
                                                                              boxes=None,
                                                                              masks=None,
                                                                              )

                self.model_constants['sparse_embeddings'] = sparse_embeddings
                self.model_constants['dense_embeddings'] = dense_embeddings

            sparse_embeddings = self.model_constants['sparse_embeddings']
            dense_embeddings = self.model_constants['dense_embeddings']

        out = self.sam_mask_decoder(image_embeddings=backbone_features,
                                    image_pe=self.sam_prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse_embeddings,
                                    dense_prompt_embeddings=dense_embeddings,
                                    multimask_output=multimask_output,
                                    repeat_image=False,  # the image is already batched
                                    high_res_features=high_res_features,
                                    )

        low_res_multimasks, ious, sam_output_tokens, object_score_logits = out

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None],
                                             low_res_multimasks,
                                             NO_OBJ_SCORE,
                                             )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(low_res_multimasks,
                                            size=(self.image_size, self.image_size),
                                            mode="bilinear",
                                            align_corners=False,
                                            )

        if multimask_output:
            low_res_masks = []
            high_res_masks = []
            sam_output_token = []
            best_ious = []
            kf_ious = []

            for obj in range(0, B):
                out = self.update_kalman_filter(obj=obj,
                                                ious=ious,
                                                low_res_multimasks=low_res_multimasks,
                                                high_res_multimasks=high_res_multimasks,
                                                sam_output_tokens=sam_output_tokens
                                                )

                _low_res_masks, _high_res_masks, _sam_output_token, _ious, _kf_ious = out

                low_res_masks.append(_low_res_masks)
                high_res_masks.append(_high_res_masks)
                sam_output_token.append(_sam_output_token)
                best_ious.append(_ious)
                kf_ious.append(_kf_ious)

            low_res_masks = torch.cat(low_res_masks).unsqueeze(1)
            high_res_masks = torch.cat(high_res_masks).unsqueeze(1)
            sam_output_token = torch.cat(sam_output_token)
            best_ious = torch.cat(best_ious)
            kf_ious = torch.cat(kf_ious)

        else:
            best_ious = ious[:, 0]
            sam_output_token = sam_output_tokens[:, 0]
            kf_ious = torch.full((2,1), float("nan"), device=self.device)

            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                # Only hard possible with gt
                assert not self.teacher_force_obj_scores_for_mem
                lambda_is_obj_appearing = object_score_logits.sigmoid()

            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr

            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                best_ious,
                kf_ious,
                )


    def update_memory_bank(self, prediction: Dict):
        """
        Adds a prediction to short-term memory and, under specific conditions, to long-term memory.

        Parameters
        ----------
        prediction : Dict
            A dictionary representing a SAM2 prediction

        Returns
        -------
        None
            This method does not return any value. It modifies self.past_frames in-place.

        Notes
        -----
        - If the short_term deque is full, the method checks whether the frame should also be
          added to the long_term memory based on an occlusion criterion.

        - The occlusion criterion evaluates the object_score_logits of the oldest short term frame. If the logits
          are greater than 0.5, it is considered not occluded and the oldest frame in short_term
          is added to long_term. 0.5 is an arbitrary threshold on a scale of 0-1 that assumes 0.5 is half-occluded.

        """

        memory_frame = {'maskmem_pos_enc': prediction['maskmem_pos_enc'],
                        'maskmem_features': prediction['maskmem_features'],
                        'obj_ptr': prediction['obj_ptr'],
                        'object_score_logits': prediction['object_score_logits'],
                        'ious': prediction['ious'],
                        'kf_ious': prediction['kf_ious']
                        }

        if len(self.past_frames['short_term']) == self.past_frames['short_term'].maxlen:
            oldest_short_term_frame = self.past_frames['short_term'].popleft()
            object_score_logits = oldest_short_term_frame['object_score_logits']

            not_occluded = (object_score_logits > 5).any()

            if not_occluded:
                self.past_frames['long_term'].append(oldest_short_term_frame)

        self.past_frames['short_term'].append(memory_frame)


    def prepare_memory_conditioned_features(self,
                                            current_vision_feats: List[torch.Tensor],
                                            current_vision_pos_embeds: List[torch.Tensor],
                                            feat_sizes: List[Tuple[int, int]],
                                            ) -> torch.Tensor:

        """
        Fuse the current frame's visual feature map with previous memory.

        Parameters
        ----------
        current_vision_feats : List[torch.Tensor]
            The feature maps of the current frame at different resolutions.

        current_vision_pos_embeds : List[torch.Tensor]
            The positional embeddings for the current frame at different resolutions.

        feat_sizes : List[Tuple]
            The spatial dimensions (height, width) of the feature maps.

        Returns
        -------
        pix_feat_with_mem: torch.Tensor
            The fused feature map with memory conditioning. Its shape is (B, C, H, W),
            where B is the batch size, C is the number of channels, and H, W are the spatial dimensions.

        """

        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device

        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        # Step 1: condition the visual features of the current frame on previous memories
        if self.past_frames['short_term'] or self.past_frames['long_term']:
            short_term = self.past_frames['short_term']
            long_term = self.past_frames['long_term']

            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []

            t_pos_and_prevs = [(0, long_term[t_pos]) for t_pos in range(0, len(long_term))]

            for t_pos in range(1, len(short_term) + 1):
                t_pos_and_prevs.append((t_pos, short_term[t_pos - 1]))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames

                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))

                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device, non_blocking=True)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                # Temporal positional encoding
                maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1])

                to_cat_memory_pos_embed.append(maskmem_enc)

            pos_and_ptrs = []
            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                num_frames = len(long_term) + len(short_term)
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)

                for pos in range(1, max_obj_ptrs_in_encoder - len(long_term) + 1):
                    out = short_term[len(short_term)-pos]
                    pos_and_ptrs.append((pos, out['obj_ptr']))

                for pos in range(1, len(long_term) + 1):
                    out = long_term[len(long_term)-pos]
                    pos_and_ptrs.append((pos, out['obj_ptr']))

                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)

                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)

                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs and max_obj_ptrs_in_encoder > 1:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)

                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)

                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]

                else:
                    num_obj_ptr_tokens = 0

        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(curr=current_vision_feats,
                                                  curr_pos=current_vision_pos_embeds,
                                                  memory=memory,
                                                  memory_pos=memory_pos_embed,
                                                  num_obj_ptr_tokens=num_obj_ptr_tokens,
                                                  )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem


    def inference(self,
                  current_vision_feats: List[torch.Tensor],
                  current_vision_pos_embeds: List[torch.Tensor],
                  feat_sizes: List[Tuple[int, int]],
                  point_inputs: Optional[torch.Tensor],
                  mask_inputs: Optional[torch.Tensor],
                  run_mem_encoder: bool = True,
                  prev_sam_mask_logits: Optional[torch.Tensor] = None,
                  ) -> Dict[str, Any]:

        """
        Perform inference on the current frame using visual features, positional embeddings,
        and optional point or mask inputs.

        Parameters
        ----------
        current_vision_feats : list of torch.Tensor
         Visual feature maps of the current frame at different resolutions.

        current_vision_pos_embeds : list of torch.Tensor
         Positional embeddings for the current frame at different resolutions.

        feat_sizes : list of tuple of int
         Spatial dimensions (height, width) of the feature maps.

        point_inputs : torch.Tensor or None
         Point inputs used for segmentation prompts.

        mask_inputs : torch.Tensor or None
         Mask inputs used for segmentation prompts.

        run_mem_encoder : bool, optional
         Whether to run the memory encoder on the predicted masks to encode
         them into memory for future use. Default is True.

        prev_sam_mask_logits : torch.Tensor or None, optional
         Previously predicted SAM mask logits to be used as input for the SAM mask decoder.

        Returns
        -------
        dict of str to Any
         A dictionary containing:
         - "pred_masks": Predicted low-resolution masks (torch.Tensor).
         - "pred_masks_high_res": Predicted high-resolution masks (torch.Tensor).
         - "obj_ptr": Object pointers (torch.Tensor).
         - "object_score_logits": Object score logits (torch.Tensor).
         - "ious": IoU scores (torch.Tensor).
         - "kf_ious": Keyframe IoU scores (torch.Tensor).
         - "maskmem_features": Memory features for the predicted masks (torch.Tensor or None).
         - "maskmem_pos_enc": Positional encodings for the memory features (torch.Tensor or None).

         """

        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                                 for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
                                 ]
        else:
            high_res_features = None

        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)

            _, _, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs
            kf_ious = torch.full((current_vision_feats[0].shape[1],1), float("nan"), device=self.device)

        else:
            torch.cuda.synchronize()
            start = time.time()
            # fused the visual feature with previous memory features in the memory bank
            pix_feat_with_mem = self.prepare_memory_conditioned_features(current_vision_feats=current_vision_feats[-1:],
                                                                         current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                                                                         feat_sizes=feat_sizes[-1:],
                                                                         )
            torch.cuda.synchronize()
            print(f"[TIMING] prepare_memory_conditioned_features took {1000*(time.time() - start):.1f} ms")

            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits

            init_frame = not self.past_frames['short_term'] and not self.past_frames['long_term']
            multimask_output = self._use_multimask(init_frame, point_inputs)

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                sam_outputs = self.forward_sam_heads(backbone_features=pix_feat_with_mem,
                                                    point_inputs=point_inputs,
                                                    mask_inputs=mask_inputs,
                                                    high_res_features=high_res_features,
                                                    multimask_output=multimask_output,
                                                    )
            torch.cuda.synchronize()
            print(f"[TIMING] forward_sam_heads took {1000*(time.time() - start):.1f} ms")

            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits, ious, kf_ious = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits
        current_out["ious"] = ious
        current_out["kf_ious"] = kf_ious

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(current_vision_feats=current_vision_feats,
                                                                        feat_sizes=feat_sizes,
                                                                        pred_masks_high_res=high_res_masks,
                                                                        object_score_logits=object_score_logits,
                                                                        is_mask_from_pts=(point_inputs is not None),
                                                                        )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc

        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out


    def preprocess_image(self,
                         img: np.ndarray,
                         img_mean: Tuple[float] = (0.485, 0.456, 0.406),
                         img_std: Tuple[float] = (0.229, 0.224, 0.225)
                         ):

        """
        Preprocess an input image for model inference, including resizing, standardization,
        and moving the image to the specified device.

        Parameters
        ----------
        img : np.ndarray
            Input image as a NumPy array of shape (H, W, C) with values in the range [0, 255].

        img_mean : tuple of float, optional
            Mean values for each channel used for standardization. Default is (0.485, 0.456, 0.406).

        img_std : tuple of float, optional
            Standard deviation values for each channel used for standardization.
            Default is (0.229, 0.224, 0.225).

        Returns
        -------
        torch.Tensor
            Preprocessed image as a tensor of shape (1, C, H, W), moved to the specified device.

        """

        image_size = self.image_size

        # Resize
        img = cv2.resize(img, (image_size, image_size)) / 255.0

        # Convert to Tensor and shape from (H,W,C) to (C,H,W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # Standardize
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        img -= img_mean
        img /= img_std

        # Move to device
        img = img.to(self.device, non_blocking=True).float().unsqueeze(0)

        return img


    def get_image_features(self, img: torch.Tensor) -> Tuple:
        """
        Extract and process image features for the current frame, expanding them to match the number
        of objects being tracked.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor of shape (1, C, H, W).

        Returns
        -------
        features : Tuple
            A tuple containing:
            - `current_vision_feats` : torch.Tensor
                High-resolution feature maps from the backbone model.
            - `current_vision_pos_embeds` : torch.Tensor
                Positional embeddings corresponding to the feature maps.
            - `feat_sizes` : list
                A list of feature map sizes as (height, width) for each feature level.

        """

        # get feature embeddings
        backbone_out = self.forward_image(img)

        # expand the features to have the same dimension as the number of objects
        expanded_image = img.expand(self.num_objects, -1, -1, -1)
        expanded_backbone_out = {"backbone_fpn": backbone_out["backbone_fpn"].copy(),
                                 "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
                                 }

        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(self.num_objects, -1, -1, -1)

        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(self.num_objects, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features

        # we just want current_vision_feats, current_vision_pos_embeds, feat_sizes
        features = features[2:]

        return features

    def get_mask_inputs(self, mask: np.ndarray) -> torch.Tensor:
        """
        Process and prepare mask inputs for the model, resizing them if necessary
        and aligning them with the required dimensions.

        Parameters
        ----------
        mask : np.ndarray
            Input mask array of shape (height, width) for a single mask
            or (n, height, width) for multiple masks.

        Returns
        -------
        mask_inputs : torch.Tensor
            A tensor of shape (num_objects, 1, image_size, image_size) containing
            the processed mask inputs, ready for the model.

        """

        mask = torch.from_numpy(mask)

        if mask.dim() == 2:
            # Case: Single mask (height, width)
            mask = mask[None, None]  # Add batch and channel dimension -> (1, 1, height, width)

        else:
            # Case: Multiple masks (n, height, width)
            mask = mask[:, None]  # Add channel dimension -> (n, 1, height, width)

        num_masks, mask_H, mask_W = mask.shape[0], mask.shape[2], mask.shape[3]

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask = torch.nn.functional.interpolate(mask.float(),
                                                   size=(self.image_size, self.image_size),
                                                   align_corners=False,
                                                   mode="bilinear",
                                                   antialias=True,  # use antialias for downsampling
                                                   )
            mask = mask >= 0.5
            mask = mask.to(self.device, dtype=torch.bfloat16, non_blocking=True)

        mask_inputs = torch.zeros((self.num_objects, 1, mask.shape[2], mask.shape[3]),
                                  device=self.device,
                                  dtype=torch.bfloat16
                                  )

        mask_inputs[self.curr_obj_idx:self.curr_obj_idx + num_masks] = mask

        return mask_inputs

    def get_point_inputs(self, box: Optional[np.ndarray] = None, points: Optional[np.ndarray] = None) -> Dict:
        """
        Prepare point inputs and their labels for the model, based on provided boxes or points.

        Parameters
        ----------
        box : np.ndarray, optional
            An array representing bounding boxes of shape (n, 2, 2), where `n` is the number
            of boxes, and each box is defined by two corner points (top-left and bottom-right).
            If provided, `points` is ignored.

        points : np.ndarray, optional
            An array of point coordinates of shape (k, 2) or (n, k, 2), where `k` is the
            number of points per instance. Used if `box` is not provided.

        Returns
        -------
        point_inputs : Dict
            A dictionary with the following keys:
            - "point_coords": A tensor of shape (num_objects, k, 2) containing point coordinates
              scaled to the model's image size.
            - "point_labels": A tensor of shape (num_objects, l) containing point labels.

        """

        if box is not None:
            box = torch.from_numpy(box)

            if box.dim() == 2:
                # Single box (2, 2)
                box = box[None]  # Add batch dimension -> (1, 2, 2)

            # Create point and label tensors for boxes
            point = box  # Assuming the box contains point coordinates
            label = torch.tensor([2, 3], device=self.device, dtype=torch.int32).unsqueeze(0).repeat(box.shape[0], 1)

        else:
            point = torch.from_numpy(points)

            if point.dim() == 2:
                # Single point (k, 2)
                point = point[None]  # Add batch dimension -> (1, k, 2)

            label = torch.ones((point.shape[0], point.shape[1]), device=self.device, dtype=torch.int32)

        point = point.to(device=self.device, dtype=torch.float32, non_blocking=True)

        # Allocate tensors outside of the conditional block
        points = torch.zeros((self.num_objects, point.shape[1], point.shape[2]), device=self.device, dtype=torch.float32)
        labels = torch.zeros((self.num_objects, label.shape[1]), device=self.device, dtype=torch.int32)

        # Assign values to the current index
        points[self.curr_obj_idx:self.curr_obj_idx + points.shape[0]] = point
        labels[self.curr_obj_idx:self.curr_obj_idx + label.shape[0]] = label

        # Scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size

        point_inputs = {"point_coords": points, "point_labels": labels}

        return point_inputs


    @torch.inference_mode()
    def track_new_object(self,
                         img: Union[np.ndarray, torch.Tensor],
                         points: Optional[np.ndarray] = None,
                         box: Optional[np.ndarray] = None,
                         mask: Optional[np.ndarray] = None
                         ) -> Dict:

        """
        Track a new object in a given image based on input image, points, boxes, or masks.

        Parameters
        ----------
        img : Union[np.ndarray, torch.Tensor]
            Input image, either as a NumPy array (H, W, C) or a preprocessed Torch tensor
            (1, C, H, W). If given as a NumPy array, it will be preprocessed.

        points : np.ndarray, optional
            Array of point coordinates, shape (k, 2) or (n, k, 2).

        box : np.ndarray, optional
            Array of bounding boxes, shape (n, 2, 2).

        mask : np.ndarray, optional
            Array of masks, shape (n, height, width).

        Returns
        -------
        prediction : Dict[str, torch.Tensor]
            A dictionary containing:
            - "maskmem_features": Memory features for masks.
            - "maskmem_pos_enc": Positional encodings for memory features.
            - "pred_masks": Predicted masks for the new objects.
            - "obj_ptr": Object pointers for the tracked objects.
            - "object_score_logits": Object score logits.

        """

        if isinstance(img, np.ndarray):
            img_height, img_width = img.shape[0:2]
            img = self.preprocess_image(img=img)

        else:
            img_height, img_width = img.shape[-2:]

        num_new_objects = 0

        if mask is not None:
            num_new_objects += mask.shape[0]

            mask_inputs = self.get_mask_inputs(mask=mask)
            point_inputs = None

        else:
            num_new_objects += box.shape[0] if box is not None else points.shape[0]

            mask_inputs = None
            point_inputs = self.get_point_inputs(box=box, points=points)
            normalization = torch.tensor([img_width, img_height], device=self.device)
            point_inputs['point_coords'] = point_inputs['point_coords'] / normalization

        """Run tracking on a single frame based on current inputs and previous memory."""
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_image_features(img)
        prediction = self.inference(current_vision_feats=current_vision_feats,
                                    current_vision_pos_embeds=current_vision_pos_embeds,
                                    feat_sizes=feat_sizes,
                                    point_inputs=point_inputs,
                                    mask_inputs=mask_inputs,
                                    run_mem_encoder=True,
                                    prev_sam_mask_logits=None,
                                    )

        self.curr_obj_idx += num_new_objects

        self.update_memory_bank(prediction=prediction)

        return prediction

    def get_memory_time(self, function, kwags) -> tuple:

        torch.cuda.synchronize()
        start_time = time.time()
        before = torch.cuda.memory_allocated()

        output = function(**kwags)
        torch.cuda.synchronize()
        time_taken = f'{(time.time() - start_time) * 1000:.1f}'
        memory_used = f'{(torch.cuda.memory_allocated() - before)/1024**2:.2f}'

        return (output), time_taken, memory_used

    @torch.inference_mode()
    def track_all_objects(self, img: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Tracks all objects in a given image and updates the memory bank with the predicted results.

        Parameters
        ----------
        img : Union[np.ndarray, torch.Tensor]
            The input image to be processed. It can be a NumPy array (H, W, C) or a preprocessed
            torch.Tensor (1, C, H, W). If a NumPy array is provided, it will be preprocessed.

        Returns
        -------
        prediction : Dict[str, torch.Tensor]
            A dictionary containing:
            - "maskmem_features": Memory features for masks.
            - "maskmem_pos_enc": Positional encodings for memory features.
            - "pred_masks": Predicted masks for all tracked objects.
            - "obj_ptr": Object pointers for the tracked objects.
            - "object_score_logits": Object score logits.

        """

        # Prepare image for inference
        preprocess_time , preprocess_mem = 0, 0
        if isinstance(img, np.ndarray):
            (img), \
            preprocess_time, \
            preprocess_mem = self.get_memory_time(self.preprocess_image, {'img' : img})

        # Retrieve image features
        (current_vision_feats, current_vision_pos_embeds, feat_sizes), \
        image_embedding_time, \
        image_embedding_mem = self.get_memory_time(self.get_image_features, {'img' : img})

        # Inference
        (prediction), \
        inference_time, \
        inference_mem = self.get_memory_time(
            self.inference, 
            {
                "current_vision_feats": current_vision_feats,
                "current_vision_pos_embeds": current_vision_pos_embeds,
                "feat_sizes": feat_sizes,
                "point_inputs": None,
                "mask_inputs": None,
                "run_mem_encoder": True,
                "prev_sam_mask_logits": None
            }
        )
        
        _, \
        memory_bank_time, \
        memory_bank_mem = self.get_memory_time(
            self.update_memory_bank,
            {'prediction': prediction}
        )

        total_time = str(float(preprocess_time) + float(image_embedding_time) + float(inference_time) + float(memory_bank_time))
        total_mem = str(float(preprocess_mem) + float(image_embedding_mem) + float(inference_mem) + float(memory_bank_mem))

        table = [
            ["Preprocess", preprocess_time, preprocess_mem],
            ["Feature Extraction", image_embedding_time, image_embedding_mem],
            ["Inference", inference_time, inference_mem],
            ["Memory Bank",memory_bank_time,memory_bank_mem],
            ['Total', total_time, total_mem]
        ]

        headers = ["Function","Time (ms)", "Memory (MB)"]

        print()
        print(tabulate(table, headers = headers,tablefmt = "simple_grid"))
        print()

        if self.verbose:
            print(f'SAM2 Tracking: {preprocess_time * 1000:.1f}ms preprocess, '
                  f'{image_embedding_time * 1000:.1f}ms image embedding, '
                  f'{inference_time * 1000:.1f}ms inference, '
                  f'{memory_bank_time * 1000:.1f}ms memory bank'
                  f' per image at shape {img.shape}'
                  )

        return prediction
