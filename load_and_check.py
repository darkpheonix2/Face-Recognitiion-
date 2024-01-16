# importing the packages
import face_recognition
import cv2
import os
import glob
import numpy as np

#defining the class and its methods

class load_check:
    #default constructor
    def __init__(self):
        self.my_face_encodings=[]
        self.my_face_names=[]

        #For faster access resize the frame
        self.frame_resizing = 0.25
    
    # Loading the images
    def load_images_to_encode(self,image_path):

        image_path=glob.glob(os.path.join(image_path,"*.*"))
        #this will store all the images of given path
        print("Total number of images found are :" + str(len(image_path)))


        #Storing image encodings with their names

        for img_pth in image_path:
            img = cv2.imread(img_pth)

            #Since it can only read images in RGB format so converting images to RGB format
            rgb_image= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # using the name of file to give name to the image
            #getting the filename 
            fname = os.path.basename(img_pth)
            # removing the extensions from file name
            (name,ext)= os.path.splitext(fname)

            #Encoding the image
            img_encoding = face_recognition.face_encodings(rgb_image)[0]
            #there could be multiple images so getting the image at index 0

            #Storing the name and image_encoding
            self.my_face_encodings.append(img_encoding)
            self.my_face_names.append(name)

        print("Encoding Completed !!")

# Time to detect the faces
    def detect_faces(self,frame):
        #resizing the frame
        s_frame = cv2.resize(frame , (0,0) , fx = self.frame_resizing , fy=self.frame_resizing)

        # Finding faces and face encodings of current frame (image) of video and converting them to RGB format
        rgb_s_image = cv2.cvtColor( s_frame , cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_s_image)
        face_encodings = face_recognition.face_encodings(rgb_s_image,face_locations)


        # filling the face_names 
        face_names = []
        for f_encoding in face_encodings:

            #Checking for image match

            matches = face_recognition.compare_faces(self.my_face_encodings,f_encoding)
            name = "Unknown"

             # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face

            face_distances = face_recognition.face_distance(self.my_face_encodings,f_encoding)
            best_match = np.argmin(face_distances)

            if matches[best_match] :
                name = self.my_face_names[best_match]
            else :
                name = input("What is the name of individual in front of camera ?\n" )
                self.my_face_encodings.append(f_encoding)
                self.my_face_names.append(name)
                to_save=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join('C:/Nakul/coding/Projects/face recognition/my project/images',(name+".png")),frame)
                
            face_names.append(name)

        # Converting to mumpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int) , face_names

