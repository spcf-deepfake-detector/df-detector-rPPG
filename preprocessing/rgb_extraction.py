import cv2
import numpy as np
from preprocessing.roi import ROIExtractor
from typing import List, Dict, Any


class RGBExtractor:
    def __init__(self):
        self.roi_extractor = ROIExtractor()

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
            raise ValueError(f"Could not find or open the video in this path: {video_path}")

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
                    rgb_signals.append([np.nan, np.nan, np.nan])
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
                    rgb_signals.append(
                        [np.nan,np.nan,np.nan]
                    )
                    stats["invalid_bbox"] += 1
                    continue

                try:
                    rois = self.roi_extractor.extract_rois(
                        frame,
                        [x,y, box_w, box_h]
                    )
                    if not isinstance(rois, dict) or not rois:
                        rgb_signals.append([np.nan, np.nan, np.nan])
                        stats["invalid_roi_structure"] += 1
                        continue
                    

                    for name, roi in rois.items():
                        if not isinstance(roi, dict):
                            print(f"ROI DEBUG! {name}: invalid ROI structure")
                            continue

                        crop = roi.get("crop")
                        mask = roi.get("mask")

                    print(
                        f"ROI DEBUG! {frame_idx}, {name}: "
                        f"crop_shape={crop.shape if isinstance(crop, np.ndarray) else None},"
                        f"mask_shape={mask.shape if isinstance(mask, np.ndarray) else None}"
                    )
                    
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

# Process every facial region in the current frame.
                for roi_name, roi_data in rois.items():

                    roi_stats["total_checked"] += 1

                    # Validate the ROI dictionary.
                    if not isinstance(roi_data, dict):
                        roi_stats["invalid_data"] += 1
                        roi_failure_reasons.append(
                            f"{roi_name}: expected a dictionary"
                        )
                        continue

                    crop = roi_data.get("crop")
                    mask = roi_data.get("mask")

                    # Validate the crop and mask.
                    if (
                        crop is None
                        or mask is None
                        or not isinstance(crop, np.ndarray)
                        or not isinstance(mask, np.ndarray)
                        or crop.size == 0
                        or mask.size == 0
                    ):
                        roi_stats["invalid_data"] += 1
                        roi_failure_reasons.append(
                            f"{roi_name}: empty or invalid crop/mask"
                        )
                        continue

                    # Check crop and mask dimensions.
                    if (
                        crop.ndim != 3
                        or crop.shape[2] != 3
                        or mask.ndim != 2
                        or mask.shape != crop.shape[:2]
                    ):
                        roi_stats["invalid_dimensions"] += 1
                        roi_failure_reasons.append(
                            f"{roi_name}: incompatible dimensions "
                            f"crop={crop.shape}, mask={mask.shape}"
                        )
                        continue

                    # Reject crops that are too small.
                    crop_h, crop_w = crop.shape[:2]

                    if (
                        crop_w < self.min_roi_width
                        or crop_h < self.min_roi_height
                    ):
                        roi_stats["small_crop"] += 1
                        roi_failure_reasons.append(
                            f"{roi_name}: small crop {crop_w}x{crop_h}"
                        )
                        continue

                    # Select the usable pixels identified by the mask.
                    skin_pixels = mask > 0

                    if not np.any(skin_pixels):
                        roi_stats["empty_mask"] += 1
                        roi_failure_reasons.append(
                            f"{roi_name}: empty mask"
                        )
                        continue

                    # OpenCV stores color channels in BGR order.
                    b_channel = crop[:, :, 0]
                    g_channel = crop[:, :, 1]
                    r_channel = crop[:, :, 2]

                    # Calculate the average color for this region.
                    r_mean = float(np.mean(r_channel[skin_pixels]))
                    g_mean = float(np.mean(g_channel[skin_pixels]))
                    b_mean = float(np.mean(b_channel[skin_pixels]))

                    # Reject invalid numerical values.
                    if not np.all(np.isfinite([r_mean, g_mean, b_mean])):
                        roi_stats["invalid_data"] += 1
                        roi_failure_reasons.append(
                            f"{roi_name}: non-finite RGB values"
                        )
                        continue

                    # Store this region's measurements.
                    r_vals.append(r_mean)
                    g_vals.append(g_mean)
                    b_vals.append(b_mean)

                    roi_stats["usable"] += 1


                # IMPORTANT: This block is OUTSIDE the facial-region loop.
                # Append exactly one measurement for the current video frame.

                if r_vals:
                    rgb_signals.append([
                        float(np.mean(r_vals)),
                        float(np.mean(g_vals)),
                        float(np.mean(b_vals))
                    ])

                    stats["valid"] += 1

                else:
                    # No facial region provided usable RGB values.
                    rgb_signals.append([np.nan, np.nan, np.nan])

                    if not roi_failure_reasons:
                        stats["no_rgb_values"] += 1

                    elif all(
                        "empty mask" in reason
                        for reason in roi_failure_reasons
                    ):
                        stats["empty_masks_only"] += 1

                    elif all(
                        "empty or invalid crop/mask" in reason
                        or "incompatible dimensions" in reason
                        or "small crop " in reason
                        for reason in roi_failure_reasons
                    ):
                        stats["no_valid_roi"] += 1

                    else:
                        stats["no_rgb_values"] += 1

                    # Save a limited number of examples for debugging.
                    if len(failure_examples) < max_faliure_examples:
                        failure_examples.append(
                            (frame_idx, roi_failure_reasons)
                        )
                        
        finally:
            cap.release()
            
        total = len(rgb_signals)
        valid = stats["valid"]
        missing = total - valid
                    
        print("---------- RGB EXTRACTION ----------")
        print(f"Video: {video_path}")
        print(f"Frames processed: {total}")
        print(f"Valid RGB measurements: {valid}")
        print(f"Missing RGB measurements: {missing}")
        
        if total:
            print(f"Valid Percentage: {valid/total:.2%}")
        else:
            print("Valid percentage: N/A")

        print("\nFrame-level failure categories:")
        print(f"Face not detected: {stats['face_not_detected']}")
        print(f"Invalid bounding box: {stats['invalid_bbox']}")
        print(f"ROI exceptions: {stats['roi_exception']}")
        print(f"Invalid ROI structure: {stats['invalid_roi_structure']}")

        print(f"No valid ROI: {stats['no_valid_roi']}")
        print(f"Empty masks only: {stats['empty_masks_only']}")
        print(f"Other frames without RGB: {stats['no_rgb_values']}")

        print(f"ROI-level diagnostics:")
        print(f"ROIs checked: {roi_stats['total_checked']}")
        print(f"Invalid ROI data: {roi_stats['invalid_data']}")
        print(
            "Incompatible crop/mask dimensions: "
            f"{roi_stats.get('invalid_dimensions', 0)}"
        )

        print(f"Empty masks: {roi_stats['empty_mask']}")
        print(f"Small crops: {roi_stats['small_crop']}")
        print(f"Usable ROIs: {roi_stats['usable']}")
        
        if failure_examples:
            print("\nExamples of frames with no usble RGB:")
            
            for frame_idx, reasons in failure_examples:
                print(f"Frame {frame_idx}:")
                
                for reason in reasons:
                    print(f" - {reason}")
                    
        if not rgb_signals:
            return np.empty((0,3), dtype=np.float32)
            
        return np.asarray(rgb_signals, dtype=np.float32)
        
    def close(self):
        self.roi_extractor.close()
