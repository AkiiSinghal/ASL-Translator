import cv2 
import numpy as np
import svm_train as st

#Get the biggest Controur
def getMaxContour(contours,minArea=200):
    maxC=None
    maxArea=minArea
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>maxArea):
            maxArea=area
            maxC=cnt
    return maxC

def dodgeV2(image, mask):
    return cv2.divide(image, 255-mask, scale=256)
    
#Get Gesture Image by prediction
def getGestureImg(cnt,img,th1,model):
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    img_gray = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),sigmaX=0, sigmaY=0)
    imgT = dodgeV2(img_gray, img_blur)
    #imgT=img[y:y+h,x:x+w]
    imgT=cv2.bitwise_and(imgT,imgT,mask=th1[y:y+h,x:x+w])
    imgT=cv2.resize(imgT,(200,200))
    #imgTG=cv2.cvtColor(imgT,cv2.COLOR_BGR2GRAY)
    resp=st.predict(model,imgT)
    img=cv2.imread('TrainData/'+chr(int(resp[0])+64)+'_2.jpg')
    return imgT,chr(int(resp[0])+64)