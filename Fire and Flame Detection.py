import cv2
import numpy as np
 
def main():
    
    cap = cv2.VideoCapture("C:/Users/SomnathRoy/Desktop/Test.mp4")
    #cap = cv2.VideoCapture(0)
    
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    while ret:

              
        blur1 = cv2.GaussianBlur(frame1, (21, 21), 0)
        blur2 = cv2.GaussianBlur(frame2, (21, 21), 0)
        
        hsv1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
 
        upper = [27, 255, 255]
        lower = [4, 80, 83]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        
        mask1 = cv2.inRange(hsv1, lower, upper)
        mask2 = cv2.inRange(hsv2, lower, upper)
 
 
        out1 = cv2.bitwise_and(frame1, hsv1, mask=mask1)
        out2 = cv2.bitwise_and(frame2, hsv2, mask=mask2)

        cv2.imshow("Flame Detection", out1)
        #cv2.imshow("Out2", out2)

        out1=cv2.resize(out1, (640,352), interpolation = cv2.INTER_AREA)
        out2=cv2.resize(out2, (640,352), interpolation = cv2.INTER_AREA)

        d = cv2.absdiff(out1, out2)
        
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        
        ret, th = cv2.threshold( blur, 20, 255, cv2.THRESH_BINARY)
    
        dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1 )
        
        eroded = cv2.erode(dilated, np.ones((3, 3), np.uint8), iterations=1 )
        
        c, h = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        c = np.array(c)

        frame1=cv2.drawContours(frame1, c, -1, (0, 0, 255), 2)

        cv2.imshow("Original", frame2)
        cv2.imshow("Output", frame1)
        
        
        if cv2.waitKey(1) == 27: # exit on pressing ESC
            break
        
        frame1 = frame2
        ret, frame2 = cap.read()
 
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
