import cv2
import numpy as np
from preprocessing.roi import ROIExtractor
from typing import List, Dict, Any


class RGBExtractor:
    def __init__(self):
        self.roi_extractor = RGBExtractor()

        self.min_roi_width = 5
        self.min_roi_height = 5

    def process_video_frames(
            self,
            video_path: str,
            tracking_results: List[Dict[str,Any]]
    )->np.ndarray:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Couldnot find or open the video in this path: {video_path}")

        rgb_signals = []

        stats = {
            "valid": 0,
            "face_not_detected": 0,
            "invalid_bbox": 0,
            "roi_exception": 0,
            "invalid_roi_structure": 0,
            "no_valid_roi": 0,
            "empty_masks_only": 0,
            "no_rgb_values": 0,
        }

        roi_stats = {
            "total_checked": 0,
            "invalid_data": 0,
            "invalid_dimensions": 0,
            "empty_mask": 0,
            "small_crop": 0,
            "usable": 0,
        }

        failure_examples = []

        max_faliure_examples = 20

        
        try:
            for frame_idx, info in enumerate(tracking_results):

                ret, frame = cap.read()

                if not ret:
                    print(
                        f"Warning! video ended at frame {frame_idx},"
                        f"but tracking results contain "
                        f"{len(tracking_results)} entries."
                    )
                    break


                detected = info.get("detected", False)
                bbox = info.get("bbox")

                if not detected:
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )
                    stats["face_not_detected"] += 1
                    continue

                if (
                    not isinstance(bbox, (list, tuple, np.ndarray))
                    or len(bbox) !=4
                ):
                    rgb_signals.append(np.nan, np.nan, np.nan)
                    stats["invalid_bbox"] +=1
                    continue

                try:
                    x,y, box_w, box_h = map(int, bbox)
                except (TypeError, ValueError, OverflowError):
                    rgb_signals.append(
                        [np.nan,np.nan,np.nan]
                    )
                    stats["invalid_bbox"] +=1
                    continue
                
                if box_w <= 0 or box_h <= 0:

                    try:
                        rois = self.roi_extractor.extract_rois(
                            frame,
                            [x,y, box_w, box_h]
                        )
                        print(f"ROI DEBUG! Frame {frame_idx}")

                        for name, roi in rois.items():
                            print(f"{name}: {type(roi)}")
                    except Exception as exc:
                        print(f"ROI ERROR! Frame {frame_idx}: {exc}")
                        rgb_signals.append([np.nan, np.nan, np.nan])
                        stats["roi_exception"] += 1
                        continue

                    if not isinstance(rois, dict) or not rois:
                        rgb_signals.append([np.nan, np.nan, np.nan])
                        stats["invalid_roi_structure"] += 1
                        continue

                    r_vals = []
                    g_vals = []
                    b_vals = []

                    roi_failure_reasons = []

                    for roi_name, roi_data in rois.items():
                       
                        if not isinstance(roi_data, dict):
                            roi_stats["invalid_data"] +=1
                            roi_failure_reasons.append(
                                f"{roi_name}: expected a dectionary"
                            )
                            continue

                        crop = roi_data.get("crop")
                        mask = roi_data.get('mask')

                        if (crop is None or mask is None or not isinstance(crop, np.ndarray) or not isinstance(mask, np.ndarray) or crop.size == 0 or mask.size == 0):
                            roi_stats["invalid_data"] += 1
                            roi_failure_reasons.append(
                                f"{roi_name}: empty or invalid crop/mask"
                            )
                            continue
                        if (crop.ndim !=3 or crop.shape[2] !=3 or mask.ndim !=2 or mask.shape != crop.shape[:2]):
                            roi_stats["invalid_dimensions"] += 1

                            roi_failure_reasons.append(
                                f"{roi_name}: incompatible dimensions "
                                f"crop={crop.shape}, mask={mask.shape}"
                            )
                            continue

                        crop_h, crop_w = crop.shape[:2]

                        if(
                            crop_w < self.min_roi_width
                            or crop_h < self.min_roi_height
                        ):
                            roi_stats["small_crop"] += 1

                            roi_failure_reasons.append(
                                f"{roi_name}: small crop "
                                f"{crop_w}x{crop_h}"
                            )

                        skin_pixels = mask > 0

                        if not np.any(skin_pixels):
                            roi_stats["empty_mask"] += 1
                            roi_failure_reasons.append(
                                f"{roi_name}: empty mask"
                            )
                            continue
                        b_channel = crop[:, :, 0]
                        g_channel = crop[:, :, 2]
                        r_channel = crop[:, :, 2]

                        r_mean = float(
                            np.mean(r_channel[skin_pixels])
                        )
                        g_mean = float(
                            np.mean(g_channel[skin_pixels])
                        )
                        r_mean = float(
                            np.mean(r_channel[skin_pixels])

                        )

                        r_vals.append(r_mean)
                        g_vals.append(g_mean)
                        b_vals.append(g_mean)

                        roi_stats["usable"] += 1

                        #part22 and on going still