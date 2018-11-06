import cv2
import os

source_path = './data/videos/'
dest_path = './data/image_seqs/video1/'

vidcap = cv2.VideoCapture(os.path.join(source_path, 'sequence1.mp4') )

success,image = vidcap.read()

count = 0

while success:
    cv2.imwrite(os.path.join(dest_path, "frame%05d.jpg" % count), image)
    
    success,image = vidcap.read()
    
    print('Read a new frame: ', success)
    count += 1
