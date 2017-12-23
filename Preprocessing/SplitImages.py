import os
import cv2
import subprocess
videoID = 1



###################################################################
output_directory = "../Datasets/Images/Video" + str(videoID) + "/"
input_filename = "../Datasets/Videos/Video" + str(videoID) + ".mp4"
output_filename_format = "Video" + str(videoID) + "_frame_%07d.png"
ffmpeg_command = ["ffmpeg", "-i",input_filename,"-r", "4",output_directory + output_filename_format]
if not os.path.exists(output_directory):
	os.makedirs(output_directory)

###################################################################
def main():
	subprocess.Popen(ffmpeg_command)
	for i in range(1,8):
		img_name = output_directory + "Video1_frame_000000" + str(i) + ".png"
		img = cv2.imread(img_name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cv2.namedWindow('image',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('image', 320,180)
		cv2.imshow("image",img)
		cv2.waitKey(1000)
if __name__ == '__main__':
	main()