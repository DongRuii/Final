# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import torch.nn as nn
import os
from torchvision import datasets
import pandas as pd
import time

from PIL import Image

from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt





#dataframe显示所有行和列
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

#from pytorch2keras.converter import pytorch_to_keras
#from keras.applications import imagenet_utils
# th architecture to use
arch = 'resnet18'
VFname = 'TLC00013'
# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
model1 = torch.load('/Users/rui/PycharmProjects/test1/Final/model.pth')
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

model.eval()
#print(model)
num_fc_in = model.fc.in_features
#model.fc = nn.Linear(num_fc_in, 10)
#print(model)

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

isstateList = []
f = open('/Users/rui/desktop/label.csv','r')
res = f.readlines()[1:]
for i in res:
    i_str = i.replace('\n','')
    ilist = i_str.split(',')
    isstateList.append(ilist)
df1 = pd.DataFrame(isstateList)
df1.columns = ['labelnumber', 'labelname','statenumber']
df1['labelnumber'] = df1['labelnumber'].astype('int')
df1 = df1.set_index('labelnumber')
df1 = df1.sort_index(ascending= True)


path = "data1"
data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),   ##imageFOLDER 返回的是一个list，这里的写法是字典的形式
                                      transform=centre_crop)
              for x in ["train", "val"]
}
classes = data_image["train"].classes # 按文件夹名字分类
classes_index = data_image["train"].class_to_idx

#output star label find_data is series findlist is list
#Cofile 不等于0的
#find_data = df1[df1['statenumber']=='2']['labelname']
Cofile = df1[df1['statenumber']!='0']['labelname']
findlist = Cofile.tolist()
#classes = tuple(findlist)
print(classes)
'''
# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
'''
# load the test image
img_name = '460.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    os.system('wget ' + img_url)
'''
img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

print('{} prediction on {}'.format(arch,img_name))

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
'''
resultList = []
listName = []
#返回所有文件中的图片名字 即时间戳
def GetImageName(dir):

    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.jpg':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName


#把图片名字从string变成int格式
def IntList():
    for i in range(len(listName)):
        resultList.append(int(listName(i)))
#flist是每一行 【时间戳，时间，s1,p1,s2,p2,s3,p3,s4,p4,s5,p5，label]
#Flist是所有时间戳flist的集合
flist = []
Flist = []
#读取每个图片 返回它预测的前五名 添加到resultlist里
#这里的resultlist只是一个图片和五个结果 因为底下for循环把resultlist重置了
def placeReco(image):

    # load the test image
    img_name = 'video/' + VFname + '/'+image+'.jpg'


    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))
    timeArray = time.strptime(image, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
   # timeStamp = image
   # timeArray = image.strftime("%Y-%m-%d %H:%M:%S")
    flist.append(int(timeStamp))
    flist.append(image)


   # print(timeStamp)

   # print(image)
    # forward pass
    logit = model1.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    resultList.append(int(timeStamp))
    #append了label的名字
    flist.append(classes[idx[0]])
    #print('{} prediction on {}'.format(arch, img_name))
    # output the prediction
    for i in range(0, 5):
      #  print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        res = []
        r = round(float(probs[i]),3)
        res.append(r)
        flist.append(r)
        res.append(classes[idx[i]])
        flist.append(classes[idx[i]])

        resultList.append(res)





img_path = "/Users/rui/PycharmProjects/test1/Final/12.jpg"


filePath = '/Users/rui/PycharmProjects/test1/Final/video/' + VFname
GetImageName(filePath)
print(listName)
#print(resultList)
#finallist是最后的list
finalList = []
for item in listName:
    placeReco(item)

   # print(resultList)
    finalList.append(resultList)
    Flist.append(flist)
    resultList=[]
    flist = []
#逐行输出
#for var in Flist:
#    print(var)
#print(finalList)

df = pd.DataFrame(Flist)
df.columns = ['timestamp', 'time','label', 'score1', 'place1', 'score2', 'place2', 'score3', 'place3','score4','place4','score5','place5']
#print(Flist)
df = df.infer_objects()
df = df.set_index('timestamp')
df = df.sort_index(ascending= True)
print (df)
save_path = "/Users/rui/desktop/timeplace13.csv"
df.to_csv(save_path,sep=',',index=True,header=True)

