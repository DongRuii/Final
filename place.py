from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import torch.nn as nn
import os
import datetime
from torchvision import datasets
import pandas as pd
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image

from PIL import Image
import torch
print(datetime.datetime.now())
arch = 'resnet18'
VFname = 'Users/rui/desktop/testset'
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
num_fc_in = model.fc.in_features

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
file_name = 'categories_places365.txt'
root = '/Users/rui/desktop/testset/'
items = os.listdir(root)



classes = list()
path = "data1"
data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),   ##imageFOLDER 返回的是一个list，这里的写法是字典的形式
                                      transform=centre_crop)
              for x in ["train", "val"]
}
classes = data_image["train"].classes # 按文件夹名字分类
classes_index = data_image["train"].class_to_idx
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
for item in items:
    if os.path.splitext(item)[1] == '.jpg':
        img = Image.open(root + item)
        input_img = V(centre_crop(img).unsqueeze(0))





        # forward pass
        logit = model1.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print('{} prediction on {}'.format(arch, item))

        # output the prediction
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

print(datetime.datetime.now())




