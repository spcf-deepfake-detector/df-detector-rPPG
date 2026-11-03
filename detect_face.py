import cv2
import os

video_path = "/home/macs/How I Make Realistic AI Videos with Seedance 2.5 - Full Guide [2hJsvDSPWYM].mp4"

print("Checking:", video_path)

if not os.path.exists(video_path):
    print("File does not exist!")
    exit()

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("OpenCV cannot open this video!")
    exit()

print("Video successfully opened!")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total frames:", frames)

video.release()