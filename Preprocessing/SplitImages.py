import os
import cv2
import subprocess

videoID = 1



###################################################################
videoID = str(videoID)
output_directory = "../Datasets/Images/Face/"
input_filename = "../Datasets/Videos/Video" + videoID + ".mp4"
ffmpeg_rename_format = "Face_frame_%02d.png"
output_filename_format = "" 
ffmpeg_command = ["ffmpeg", "-i",input_filename,"-r", "10",output_directory + ffmpeg_rename_format]
if not os.path.exists(output_directory):
	os.makedirs(output_directory)

###################################################################
def main():
	subprocess.Popen(ffmpeg_command).wait()
	index = 1
	img_filename = output_directory + "Face_frame_" + str(index).zfill(2) + ".png"
	while(os.path.isfile(img_filename)):
		img_filename = output_directory + "Face_frame_" + str(index).zfill(2) + ".png"
		img = cv2.imread(img_filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(90,150))
		index += 1
		print(img_filename)
		cv2.imwrite(img_filename,img) #Can alter compression level here (longer preprocessing but smaller files)


		
if __name__ == '__main__':
	main()