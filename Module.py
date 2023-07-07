import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self , mode=False , upBody =False ,smooth=True , detectionCon=0.5 ,trackingCon =0.5 ):
        self.mode =mode
        self.upBody = upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackingCon=trackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode ,1,self.upBody,self.smooth, self.detectionCon, self.trackingCon)

    def findPose(self,img ,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img ,self.results.pose_landmarks , self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self,img,draw=True):
        lmList=[]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks.landmark:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print(id ,lm)
                cx ,cy =int(lm.x*w) , int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img , (cx,cy) ,5 ,(255,0,0) ,cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('4.mp4')
    pTime = 0;
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (720, 480))
        detector.findPose(img, draw=False)
        lmList = detector.getPosition(img)
        #print(lmList[14])
        cv2.circle(img , (lmList[1][1], lmList[1][2]) , 15 , (256,200,0), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()