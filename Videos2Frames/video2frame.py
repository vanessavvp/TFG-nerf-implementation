import cv2
import os

cam = cv2.VideoCapture('./inputVideos/mic-nerf.mp4')
framesDirectoryName = 'dataset-3'
try:
	if not os.path.exists(framesDirectoryName):
		os.makedirs(framesDirectoryName)

except OSError:
	print ('Error: Creating directory ', framesDirectoryName)

currentframe = 0

while(True):
	ret,frame = cam.read()
	if ret:
		name = './' + framesDirectoryName + '/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name)
		cv2.imwrite(name, frame)
		currentframe += 1
	else:
		break

cam.release()
cv2.destroyAllWindows()
