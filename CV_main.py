import cv2
from load_and_check import load_check
import matplotlib.pyplot as plt

# Encoding faces from folder
#Initializing
fcs = load_check()
#Loading
fcs.load_images_to_encode('C:/Nakul/coding/Projects/face recognition/my project/images')

# Turning on the camera
cap = cv2.VideoCapture(0)
# 0 means the first webcam

while True:
    _, frame = cap.read()

    #Detecting faces
    face_locations , face_names = fcs.detect_faces( frame )

    #finding image coordinates -> top left and bottom right
    # making a rectangle with coordinate
    for f_locations , f_name in zip (face_locations,face_names):
        y1 ,x1 , y2 , x2 =f_locations[0],f_locations[1],f_locations[2],f_locations[3]

        #adding a text
        cv2.putText(frame,f_name,(x2,y1 -10) , cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,255,255),2)
        #making a rectangle
        cv2.rectangle(frame,(x1,y1),(x2,y2),(200,0,0),4)
    cv2.imshow("Web Camera" , frame)

    #if you want to save those pictures for fun
    # plt.imshow( frame)
    # plt.imshow( frame)
    # plt.show()
    # plt.show()


    #press escape key to quit webcam
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()