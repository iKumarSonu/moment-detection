import cv2
import time
import imutils

cam = cv2.VideoCapture(2) #initialize camera // if it did not initialize your camera, change the number between 0 to 10
time.sleep(1) #1 second delay

firstFrame=None #initializig there are no object
area = 500 #threshold

while True:
    _,img = cam.read() #read frame from camera
    text = "Normal"
    img = imutils.resize(img, width=500) #resize the image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert color to gray
    gaussianImg = cv2.GaussianBlur(grayImg,(21,21),0) #smoothening
    if firstFrame is None:
        firstFrame = gaussianImg
        continue
    imgDiff = cv2.absdiff(firstFrame,gaussianImg) #subtracting current frame with first frame
    threshImg = cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2) #remove holes
    cnts = cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #covering the moving object
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)
        text = "Moving Object detected"
    print(text)
    cv2.putText(img,text,(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
    cv2.imshow("cameraFeed",img)
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
