
from keras.applications import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
import numpy as np
import os
import datetime

print(datetime.datetime.now())

model = InceptionResNetV2(weights='imagenet', include_top=True)
root = '/Users/rui/desktop/testset'
items = os.listdir(root)
for item in items:
    if os.path.splitext(item)[1] == '.jpg':
        img_path = os.path.join(root, item)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)

        print('Predicted:',item, imagenet_utils.decode_predictions(features, top=5)[0])

#img_path = '/Users/rui/desktop/selection/875.jpg'
#img_path = '/Users/rui/desktop/2.jpg'
print(datetime.datetime.now())
