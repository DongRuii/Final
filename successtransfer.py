import torch
from torch.autograd import Variable as V
import torchvision
import torchvision.models as models
from torchvision import transforms as trn
from torchvision import datasets
from torch.nn import functional as F
import torch.nn as nn
import os
import pandas as pd
import shutil
from PIL import Image
import tqdm
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch.optim as optim
#from pytorch2keras.converter import pytorch_to_keras
#from keras.applications import imagenet_utils
# th architecture to use
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
print(findlist)
for item in findlist:
    source_pathT = os.path.abspath(r'/Users/rui/PycharmProjects/test1/Final/data_256/'+item)
    target_pathT = os.path.abspath(r'/Users/rui/PycharmProjects/test1/Final/data1/train/'+item)
    source_pathV = os.path.abspath(r'/Users/rui/PycharmProjects/test1/Final/mess/'+item)
    target_pathV = os.path.abspath(r'/Users/rui/PycharmProjects/test1/Final/data1/val/'+item)

    if not os.path.exists(target_pathT):
        # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(target_pathT)

    if os.path.exists(target_pathT):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(target_pathT)
    if not os.path.exists(target_pathV):
        # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(target_pathV)

    if os.path.exists(target_pathV):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(target_pathV)
    shutil.copytree(source_pathT, target_pathT)
    shutil.copytree(source_pathV, target_pathV)






DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
EPOCH = 50
BTACH_SIZE = 32
path = "data1"
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),   ##imageFOLDER 返回的是一个list，这里的写法是字典的形式
                                      transform=centre_crop)
              for x in ["train", "val"]
}
print(data_image)


all_data = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=4,
                                                    shuffle=True)
                     for x in ["train", "val"]}

data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=4,
                                                    shuffle=True)
                     for x in ["train", "val"]}

'''
all_data =  torchvision.datasets.ImageFolder(
        root=train_root,
        transform=train_transform
    )
'''
#train_data , vaild_data= torch.utils.data.random_split(all_data,[int(0.8*len(all_data)),len(all_data)-int(0.8*len(all_data))])
'''
train_set = torch.utils.data.DataLoader(
    train_data,
    batch_size=BTACH_SIZE,
    shuffle=True
)

test_set = torch.utils.data.DataLoader(
    vaild_data,
    batch_size=BTACH_SIZE,
    shuffle=False
)

'''
train_set = {x: torch.utils.data.DataLoader(dataset = data_image[x],batch_size=4,shuffle=True)for x in ["train"]}
test_set = {x: torch.utils.data.DataLoader(dataset = data_image[x],batch_size=4,shuffle=True)for x in ["val"]}


# 检查电脑GPU资源
use_gpu = torch.cuda.is_available()
print(use_gpu)  # 查看用没用GPU，用了打印True，没用打印False

classes = data_image["train"].classes # 按文件夹名字分类
classes_index = data_image["train"].class_to_idx # 文件夹类名所对应的链值
print(classes) # 打印类别
print(classes_index)

# 打印训练集，验证集大小
print("train data set:", len(data_image["train"]))
print("val data set:", len(data_image["val"]))

arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

model.eval()
print(model)

for parma in model.parameters():
    parma.requires_grad = False

num_fc_in = model.fc.in_features
model.fc = nn.Linear(num_fc_in, len(findlist))
print(model)



X_train,y_train = next(iter(all_data["train"]))
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
img = torchvision.utils.make_grid(X_train)
img = img.numpy().transpose((1,2,0))
img = img*std + mean

print([classes[i] for i in y_train])
plt.imshow(img)
#plt.show()

#for index, parma in enumerate(model.classifier.parameters()):
#    if index == 6:
#        parma.requires_grad = True

if use_gpu:
    model = model.cuda()
print(parma)


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口
  #  model = 'ResNet34'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '/Users/rui/PycharmProjects/test1/Final/data1/train/'  # 训练集存放路径
    test_data_root = '/Users/rui/PycharmProjects/test1/Final/data1/val'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 48  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

opt = DefaultConfig()
# 定义代价函数
cost = torch.nn.CrossEntropyLoss()
# 定义优化器
#optimizer = torch.optim.Adam(model.classifier.parameters())
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9)
lr = opt.lr
# 再次查看模型结构
print(model)

### 开始训练模型
n_epochs = 1
for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader_image[param]:
            batch += 1
            X, y = data
            if use_gpu:
                X, y = V(X.cuda()), V(y.cuda())
            else:
                X, y = V(X), V(y)

            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)

            loss = cost(y_pred, y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            # running_loss += loss.data[0]
            running_correct += torch.sum(pred == y.data)
            if batch % 5 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

        epoch_loss = running_loss / len(data_image[param])
        epoch_correct = 100 * running_correct / len(data_image[param])

        print("{} Loss:{:.4f}, Correct:{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))

torch.save(model, '/Users/rui/PycharmProjects/test1/Final/model.pth')

'''
criteration = nn.CrossEntropyLoss()


def train(model, device, dataset, optimizer, epoch):
    model.train()
    correct = 0
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criteration(output, y)
        loss.backward()
        optimizer.step()

    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch, loss, correct, len(dataset),
                                                                 100 * correct / len(dataset)))


def vaild(model, device, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = nn.CrossEntropyLoss(output, y)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    print(
        "Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss, correct, len(dataset), 100. * correct / len(dataset)))


#model = torchvision.models.resnet50(pretrained=True)
#model.fc = nn.Sequential(
 #   nn.Linear(2048, 2)
#)
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)

for epoch in range(1, EPOCH + 1):
    train(model, DEVICE, train_set, optimizer, epoch)
    vaild(model, DEVICE, test_set)
'''

model1 = torch.load('/Users/rui/PycharmProjects/test1/Final/model.pth')
img_name = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/2020-08-02 20:17:52.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    os.system('wget ' + img_url)

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model1.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

print('{} prediction on {}'.format(arch,img_name))

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))