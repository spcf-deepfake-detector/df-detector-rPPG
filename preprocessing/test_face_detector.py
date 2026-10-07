from face_detection import FaceDetector

video_path = "face_video.mp4"  # Replace with your test video path

detector = FaceDetector(min_detection_confidence=0.5, model_selection=0)

result = detector.process_video(video_path)

print(f"FPS: {result['fps']}")
print(f"Total Frames: {result['total_frames']}")
print(f"Detection Rate: {result['detection_rate']:.2f}")

print("\n===== Frame Detections =====")
for detection in result["detections"][:10]:
    print(f"Frame {detection['frame_idx']}: BBox={detection['bbox']}, Confidence={detection['confidence']:.2f}, Detected={detection['detected']}")

detector.close()