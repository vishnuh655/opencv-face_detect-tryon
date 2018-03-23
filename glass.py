import cv2
import numpy as np

#Load XML classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Capture video from device 0
cap = cv2.VideoCapture(-1)

#Read overlay image
glass_img = cv2.imread('glass_image.jpg')

while True:
    #Read captured Video
    ret, frame = cap.read()

    #Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    centers=[]

    #Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            centers.append((x+int(ex+0.5*ew), y+int(ey+0.5*eh)))

        if len(centers)>0:
            glasses_width = 2.16*abs(centers[1][0]-centers[0][0])
            overlay_img = np.ones(frame.shape,np.uint8)*255
            h,w = glass_img.shape[:2]
            scaling_factor = glasses_width/w
            overlay_glasses = cv2.resize(glass_img,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)

            x = centers[0][0] if centers[0][0]<centers[1][0] else centers[1][0]

            x -=  0.26*overlay_glasses.shape[1]
            y +=  0.85*overlay_glasses.shape[0]

            h,  w = overlay_glasses.shape[:2]
            overlay_img[int(y):int(y+h),  int(x):int(x+w)]  = overlay_glasses

            gray_glasses  = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
            ret,  mask  = cv2.threshold(gray_glasses, 110,  255,  cv2.THRESH_BINARY)
            mask_inv  = cv2.bitwise_not(mask)
            temp  = cv2.bitwise_and(frame,  frame,  mask=mask)
            temp2 = cv2.bitwise_and(overlay_img,  overlay_img,  mask=mask_inv)
            final_frame = cv2.add(temp, temp2)
 
    
            cv2.imshow('frame',final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows