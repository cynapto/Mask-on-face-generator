import dlib
import imutils
import os
import cv2
from imutils import face_utils
import math
import numpy as np



input1='folder'#specify it, and modify the the folder or image or video path
mask_image_path='1_1.jpg'

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha_inv * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha * img[y1:y2, x1:x2, c])
        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

if input1=="video":
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        frame = cv2.flip(frame,1)
        # load the image and perform some operations on it
        
        face_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boundary = detector(face_Gray, 1)
        for (index, rectangle) in enumerate(boundary):
            shape = predictor(face_Gray, rectangle)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rectangle)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
            cv2.putText(frame, "Face {}".format(index + 1), (x - 10, y - 10), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
            

            x1 = shape[np.argmin(shape[:,0])][0]
            x2 = shape[np.argmax(shape[:,0])][0]
            y1 = shape[np.argmin(shape[:,1])][1]
            y2 = shape[np.argmax(shape[:,1])][1]
            
            s_img = cv2.imread(mask_image_path)            
            s_img = cv2.resize(s_img, ((x2 - x1), (y2 - y1)))
            overlay_image_alpha(frame, s_img, (shape[0][0], shape[0][1]), s_img[:, :, 0]/ 255.0)
    
        cv2.imshow("detected face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if input1=="image":
    frame=cv2.imread(r"C:/Users/User DT-017/Desktop/1.jpg")#input face image path
# =============================================================================
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# =============================================================================
    
    # load the image and perform some operations on it
    
    face_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boundary = detector(face_Gray, 1)
    for (index, rectangle) in enumerate(boundary):
        shape = predictor(face_Gray, rectangle)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rectangle)
        x1 = shape[np.argmin(shape[:,0])][0]
        x2 = shape[np.argmax(shape[:,0])][0]
        y1 = shape[np.argmin(shape[:,1])][1]
        y2 = shape[np.argmax(shape[:,1])][1]
        
        s_img = cv2.imread(mask_image_path)
        
        s_img = cv2.resize(s_img, ((x2 - x1), (y2 - y1)))
    
        overlay_image_alpha(frame, s_img, (shape[0][0], shape[0][1]), s_img[:, :, 0]/ 255.0)
    
    cv2.imshow("detected face", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if input1=="folder":
    folder_path=r"C:\prdp\dataset\Images (all)"
    write_folder_path=r"C:\prdp\dataset\face with mask"
    folder_file_list=os.listdir(folder_path)
    for file1 in folder_file_list:
        file1_path=os.path.join(folder_path,file1)
        frame=cv2.imread(file1_path)
        frame_shape=frame.shape
    
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')     
        face_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boundary = detector(face_Gray, 1)
        for (index, rectangle) in enumerate(boundary):
            shape = predictor(face_Gray, rectangle)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rectangle)
            x1 = shape[np.argmin(shape[:,0])][0]
            x2 = shape[np.argmax(shape[:,0])][0]
            y1 = shape[np.argmin(shape[:,1])][1]
            y2 = shape[np.argmax(shape[:,1])][1]
            
            s_img = cv2.imread(mask_image_path)
            
            s_img = cv2.resize(s_img, ((x2 - x1), (y2 - y1)))
            overlay_image_alpha(frame, s_img, (shape[0][0], shape[0][1]), s_img[:, :, 0]/ 255.0)
        
        cv2.imshow("detected face", frame)
        write_file1_path=os.path.join(write_folder_path,file1)
        cv2.imwrite(write_file1_path,frame)
        cv2.waitKey(0)
cv2.destroyAllWindows()