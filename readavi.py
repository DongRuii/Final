import cv2
import os, sys
import time
from datetime import timedelta, datetime

#要提取视频的文件名，隐藏后缀
sourceFileName='TLC00013'

#在这里把后缀接上
video_path = os.path.join("", "", sourceFileName+'.AVI')
s = os.stat(video_path)
#s = os.stat(sourceFileName)
created = s.st_birthtime
created_tuple = time.localtime(created)
# created = datetime(*created_tuple[:7])
#created是第一个文件的时间戳
#created_str是字符串的时间戳
created = datetime.fromtimestamp(created)
created_str = time.strftime("%Y%m%d%H%M", created_tuple)
print(created)
times=0
#提取视频的频率，每30帧提取一个
frameFrequency=1
#输出图片到当前目录video文件夹下
outPutDirName='video/'+sourceFileName+'/'
if not os.path.exists(outPutDirName):
    #如果文件目录不存在则创建目录
    os.makedirs(outPutDirName)
camera = cv2.VideoCapture(video_path)
while True:
    times+=1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency==0:
        cv2.imwrite(outPutDirName + str(times)+'.jpg', image)
        print(outPutDirName + str(times)+'.jpg')
print('finished')
img_files = os.listdir(outPutDirName)
img_files = [x for x in img_files if x.endswith('.jpg')]
img_files = [x for x in img_files if not x.startswith('._')]

for img_f in sorted(img_files):
    n_str = img_f.replace('.jpg', '')
    # print("%s %s" % (img_f, n_str))
    n = int(n_str)
    nn = n / 60

    taken = created + timedelta(seconds= n - frameFrequency)
    # taken_tuple = time.localtime(taken)
    taken_str = taken.strftime("%Y-%m-%d %H:%M:%S")
  #  timeArray = taken.strftime(taken_str, "%Y-%m-%d %H:%M:%S")
  #  timeStamp = int(time.mktime(taken_str))
    # print("%s %5d %s" % (img_f, n, taken_str))
    os.rename(
        os.path.join(outPutDirName, img_f),
        #如果是taken_str说明是用时间做名字 如果是用taken是用时间戳
        os.path.join(outPutDirName, taken_str + '.jpg'),
    )

camera.release()
