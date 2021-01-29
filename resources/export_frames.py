import cv2
import argparse

parser = argparse.ArgumentParser(description = "export video to individual frames in .jpg")
parser.add_argument("-f", "--file", help="provide path to video file (.mp4)", required=True)
args = parser.parse_args()


if args.file and args.file.endswith('.mp4'):
	capture_video = cv2.VideoCapture(args.file)
	success,img = capture_video.read()
	cnt = 0

	while success:
		cv2.imwrite("data/all_frames/frame_%04d.jpg" % cnt, img)     # save frame as JPEG file      
		success,img = capture_video.read()
		print('Read and exported frame #%d: ' % cnt, success)
		cnt += 1
	print("{} frames exported.".format(cnt))
else:
	print("File doesn't exist or has wrong format.")
