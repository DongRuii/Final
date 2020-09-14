from keras.applications.vgg16 import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import imagenet_utils
import numpy as np
import os
import datetime

model = VGG16(weights='imagenet', include_top=True)
model1 = VGG19(weights='imagenet', include_top=True)
model2 = ResNet50(weights='imagenet', include_top=True)


print(datetime.datetime.now())
#img_path = '/Users/rui/desktop/selection/809.jpg'

#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

#features = model.predict(x)



root = '/Users/rui/desktop/test'
items = os.listdir(root)
for item in items:
    if os.path.splitext(item)[1] == '.jpg':
        img_path = os.path.join(root, item)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model2.predict(x)

        print('Predicted:',item, imagenet_utils.decode_predictions(features, top=5)[0])

#img_path = '/Users/rui/desktop/selection/875.jpg'
#img_path = '/Users/rui/desktop/2.jpg'
print(datetime.datetime.now())


#print('Predicted:', imagenet_utils.decode_predictions(features,top=5)[0])
