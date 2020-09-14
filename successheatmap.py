# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torchvision import transforms as trn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import os
import torch
import time
import pandas as pd
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
model1 = torch.load('/Users/rui/PycharmProjects/test1/Final/model.pth')
#以上是model的导入 不用看

checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

# input image
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
   # net = models.resnet18(pretrained=True)
    net = model
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

#以上是不同库的结构导入不用看


# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'''
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])
'''
listName = []
def GetImageName(dir):

    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.jpg':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName

#文件读取尝试
img_path = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007' #图像读取文件夹
save_map_path = '/Users/rui/PycharmProjects/test1/Final/video/Feature' #要保存的文件夹
GetImageName(img_path)
filelist = os.listdir(img_path) #打开对应文件夹
total_num = len(filelist)  #得到文件夹中图像的个数
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

#output star label find_data is series findlist is list
#Cofile 不等于0的
#find_data = df1[df1['statenumber']=='2']['labelname']
Cofile = df1[df1['statenumber']!='0']['labelname']
findlist = Cofile.tolist()
classes = tuple(findlist)
for file in listName:
    #这个是每个图片的文件路径
    imagejpg = file + '.jpg'
    img_name = os.path.join(img_path, imagejpg)
    img= Image.open(img_name)
    img_variable = Variable(centre_crop(img).unsqueeze(0))
    logit = net(img_variable)
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    Cofile = df1[df1['statenumber'] != '0']['labelname']
    findlist = Cofile.tolist()
    classes = tuple(findlist)
  #  classes = list()
  #  with open(file_name) as class_file:
  #      for line in class_file:
  #          classes.append(line.strip().split(' ')[0][3:])
   # classes = tuple(classes)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    for i in range(0, 5):
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[i]])
        img = cv2.imread(img_name)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        #不带后缀的图片名称（也就是时间）

       # fileName = os.path.splitext(file)[0] + '\n'
        timeArray = time.strptime(file, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        newmappath = save_map_path + '/' + str(timeStamp) + '-' + str(i) + '.jpg'
        cv2.imwrite(newmappath, result)


'''
    #img_pil = '1001.jpg'
img_name = '1001.jpg'
#img_tensor = preprocess(img_pil)
#img_variable = Variable(img_tensor.unsqueeze(0))

img = Image.open(img_name)
img_variable = Variable(centre_crop(img).unsqueeze(0))

logit = net(img_variable)

# download the imagenet category list
#classes = {int(key):value for (key, value)
#          in requests.get(LABELS_URL).json().items()}
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
#idx 代表了第几个feature，0是第一个
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread('1001.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM3.jpg', result)
'''