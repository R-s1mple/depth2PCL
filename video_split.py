import cv2
import numpy as np


cap = cv2.VideoCapture("VID_20220109_162935.mp4")
c = 1
frameRate = 10  # 帧数截取间隔（每隔10帧截取一帧）
 
while(True):
    ret, frame = cap.read()
    frame
    if ret:
        if(c % frameRate == 0):
            print("开始截取视频第：" + str(c) + " 帧")
            # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
            frame = np.rot90(frame,-1)
            cv2.imwrite("capture_image/" + str(c) + '.jpg', frame)  # 这里是将截取的图像保存在本地
        c += 1
        #cv2.waitKey(0)
    else:
        print("所有帧都已经保存完成")
        break
cap.release()