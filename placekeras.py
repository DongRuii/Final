# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
#from pytorch2keras.converter import pytorch_to_keras
#from keras.applications import imagenet_utils
# th architecture to use
arch = 'resnet18'


import numpy as np
import torch
from torch.autograd import Variable
#from pytorch2keras.converter import pytorch_to_keras
import torchvision.models as models
import torch.onnx
import onnx
from keras.models import load_model

pytorch_model = '/path/to/pytorch/model'
keras_output = '/Users/rui/PycharmProjects/test1/Final/model.hdf5'
#onnx.convert(pytorch_model, keras_output)
model = load_model(keras_output)
#preds = model.predict(x)

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

onnx.convert(model, keras_output)
modelk = load_model(keras_output)
print(modelk)

input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
dummy_input = Variable(torch.rand(1, 3, 224, 224))
print(input_np.shape)
input_var = Variable(torch.FloatTensor(input_np))
#k_model = pytorch_to_keras(model, input_var, [(3, 224, 224,)], verbose=True, change_ordering=True)
#k_model = pytorch_to_keras().pytorch_to_keras(model,dummy_input,[(3,224,224,)],verbose=True)

#k_model.summary()
#k_model.summary()
#print(k_model)


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

# load the test image
img_name = '12.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    os.system('wget ' + img_url)

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)
'''
k_model = pytorch_to_keras(model, input_img, [(3, 224, 224,)], verbose=True, change_ordering=True)
k_model.summary()
k_model.save('my_model.h5')
'''
from keras.models import load_model
import tensorflow as tf
model1 = load_model('my_model.h5',custom_objects={"tf": tf})

features = model1.predict(img_name)
#print('Predicted:', imagenet_utils.decode_predictions(features,top=5)[0])
print('{} prediction on {}'.format(arch,img_name))

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

resultList = []
listName = []
'''
def GetImageName(dir):

    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.jpg':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName

filePath = '/Users/rui/PycharmProjects/test1/Final/video/TLC00004'
GetImageName(filePath)
print(listName)

def IntList():
    for i in range(len(listName)):
        resultList.append(int(listName(i)))


def placeReco(image):

    # load the test image
    img_name = 'video/TLC00004/'+image+'.jpg'


    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    resultList.append(image)
    #print('{} prediction on {}'.format(arch, img_name))
    # output the prediction
    for i in range(0, 5):
      #  print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        res = []
        r = round(float(probs[i]),3)
        res.append(r)
        res.append(classes[idx[i]])

        resultList.append(res)

#print(resultList)
finalList = []
for item in listName:
    placeReco(item)

   # print(resultList)
    finalList.append(resultList)
    resultList=[]
for var in finalList:
    print(var)
#print(finalList)
'''
