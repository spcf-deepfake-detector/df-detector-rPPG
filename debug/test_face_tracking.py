import os
import cv2
import argparse

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_VIDEO = (
    "dataset/train/fake/"
    "Screen Recording 2026-01-23 091908.mp4"
    
)

DEFAULT_OUTPUT = (
    "debug/output/tracking_result.mp4"
)


# ==========================================================
# DRAWING FUNCTIONS
# ==========================================================

def draw_box(
    frame,
    bbox,
    color,
    label,
    thickness=2,
):
    """Draw a bounding box and label on a video frame."""

    if not isinstance(bbox, (list, tuple)):
        return

    if len(bbox) != 4:
        return

    x, y, width, height = map(int, bbox)

    if width <= 0 or height <= 0:
        return

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(frame_width - 1, x + width)
    y2 = min(frame_height - 1, y + height)

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    text_y = max(20, y1 - 8)

    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def get_face_candidates(detection):
    """
    Get the original face candidates from the detector.

    The list order must be preserved because the tracker
    stores the selected candidate's original face index.
    """

    faces = detection.get("faces")

    if isinstance(faces, list):
        return faces

    # Support the older single-face detector format.
    if detection.get("detected", False):

        return [{
            "bbox": detection.get("bbox"),
            "confidence": detection.get("confidence", 0.0),
            "detected": True,
        }]

    return []


# ==========================================================
# FRAME ANNOTATION
# ==========================================================

def annotate_frame(
    frame,
    detection,
    tracking,
):
    """Draw detected faces, tracked target, and tracking status."""

    faces = get_face_candidates(detection)

    selected_index = tracking.get(
        "selected_face_index"
    )

    tracking_valid = tracking.get(
        "track_valid",
        False,
    )

    status = tracking.get(
        "tracking_status",
        tracking.get("status", "unknown"),
    )

    # ------------------------------------------------------
    # 1. DRAW OTHER FACES IN RED
    # ------------------------------------------------------

    for index, face in enumerate(faces):

        if not isinstance(face, dict):
            continue

        # The selected face will be drawn separately in green.
        if tracking_valid and index == selected_index:
            continue

        bbox = face.get("bbox")

        confidence = face.get(
            "confidence",
            0.0,
        )

        draw_box(
            frame=frame,
            bbox=bbox,
            color=(0, 0, 255),
            label=f"Other Face {index} ({confidence:.2f})",
            thickness=2,
        )

    # ------------------------------------------------------
    # 2. DRAW TRACKED TARGET IN GREEN
    # ------------------------------------------------------

    if tracking_valid:

        tracked_bbox = tracking.get("bbox")

        draw_box(
            frame=frame,
            bbox=tracked_bbox,
            color=(0, 255, 0),
            label="TRACKED TARGET",
            thickness=3,
        )

    # ------------------------------------------------------
    # 3. DRAW INFORMATION PANEL
    # ------------------------------------------------------

    frame_idx = tracking.get(
        "frame_idx",
        0,
    )

    candidate_count = tracking.get(
        "candidate_count",
        0,
    )

    info_lines = [
        f"Frame: {frame_idx}",
        f"Status: {status}",
        f"Faces detected: {candidate_count}",
        f"Tracking valid: {tracking_valid}",
    ]

    if selected_index is not None:

        info_lines.append(
            f"Selected face index: {selected_index}"
        )

    panel_width = 390
    panel_height = 15 + len(info_lines) * 27

    cv2.rectangle(
        frame,
        (0, 0),
        (panel_width, panel_height),
        (30, 30, 30),
        -1,
    )

    for index, text in enumerate(info_lines):

        cv2.putText(
            frame,
            text,
            (12, 30 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# ==========================================================
# VIDEO PROCESSING
# ==========================================================

def process_video(
    video_path,
    output_path,
    preview=False,
):
    """Detect, track, annotate, and save the video."""

    # ------------------------------------------------------
    # VALIDATE INPUT
    # ------------------------------------------------------

    if not os.path.isfile(video_path):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    print("\n========== FACE TRACKING VISUAL TEST ==========")

    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")

    # ------------------------------------------------------
    # STEP 1: FACE DETECTION
    # ------------------------------------------------------

    print("\n[1/4] Detecting faces...")

    with FaceDetector(
        min_detection_confidence=0.30,
        model_selection=1,
        frame_scale=1.0,#turn into 1 temporarily for testing

        max_faces=None,
        use_dual_model=False, #for testing, set to False
    ) as detector:

        detection_results = detector.process_video(
            video_path,
            multi_face=True,
            verbose=True,
        )

    detections = detection_results["detections"]

    print(
        f"\nDetection records: {len(detections)}"
    )

    if not detections:

        raise RuntimeError(
            "The detector did not return any frame records."
        )

    # ------------------------------------------------------
    # STEP 2: FACE TRACKING
    # ------------------------------------------------------

    print("\n[2/4] Tracking the initially selected face...")

    tracker = FaceTracker(
        iou_threshold=0.20,
        smoothing_alpha=0.65,
        fast_motion_threshold=0.15,
        max_center_distance=1.5,
        max_size_ratio=2.5,
        max_missed_frames=5,
        max_prediction_frames=5,
        max_target_lost_frames=30,
        debug=False,
    )

    tracked_results = tracker.track_faces(
        detections
    )

    #for temporary testing, print the tracked results
    print("\nTracked results:")
    print("First tracking Record:", tracked_results[0])

    print("\nAvailable keys in tracking record:", tracked_results[0].keys())

    if len(detections) != len(tracked_results):

        raise RuntimeError(
            "Detection and tracking record counts do not match."
        )

    print(
        f"Tracking records: {len(tracked_results)}"
    )

    # ------------------------------------------------------
    # STEP 3: OPEN ORIGINAL VIDEO
    # ------------------------------------------------------

    print("\n[3/4] Opening video for annotation...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if width <= 0 or height <= 0:

        cap.release()

        raise RuntimeError(
            "Could not determine the video's dimensions."
        )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    # Create lookup dictionaries using frame indexes.
    detection_map = {
        record["frame_idx"]: record
        for record in detections
    }

    tracking_map = {
        record["frame_idx"]: record
        for record in tracked_results
    }

    # ------------------------------------------------------
    # STEP 4: ANNOTATE VIDEO
    # ------------------------------------------------------

    print("\n[4/4] Drawing bounding boxes...")

    frame_position = 0
    frames_written = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            detection = detection_map.get(
                frame_position
            )

            tracking = tracking_map.get(
                frame_position
            )

            if detection is not None and tracking is not None:

                frame = annotate_frame(
                    frame=frame,
                    detection=detection,
                    tracking=tracking,
                )

            else:

                cv2.putText(
                    frame,
                    f"Frame {frame_position}: No tracking record",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)

            frames_written += 1

            # Optional live preview.
            if preview:

                cv2.imshow(
                    "FalseFacet Face Tracking",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nPreview stopped by user.")
                    break

            frame_position += 1

    finally:

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n========== VISUAL TEST COMPLETE ==========")

    print(f"Frames written: {frames_written}")
    print(f"Output saved: {output_path}")

    print("\nColor guide:")
    print("  GREEN = Selected target")
    print("  RED   = Other detected faces")

    print("\nNext step:")
    print("Watch the annotated video.")
    print("Check whether the green box follows the same person.")

    print("============================================\n")


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visual face detection and tracking test "
            "for FalseFacet."
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_VIDEO,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to save the annotated video.",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display the annotated video during processing.",
    )

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_path=args.output,
        preview=args.preview,
    )

    

if __name__ == "__main__":
    main()

