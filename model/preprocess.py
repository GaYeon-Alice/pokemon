import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

def remove_background(img):
    background = (0, 0, 0, 0)
    threshold = 240

    for x in range(img.width):
        for y in range(img.height):
            pixel = img.getpixel((x, y))
            # RGB 값이 모두 240 이상이면 색깔을 투명하게 변경
            if pixel[0] >= threshold and pixel[1] >= threshold and pixel[2] >= threshold:
                img.putpixel((x, y), background)

    return img

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):

        # 이미지 사이즈 조정
        img = img.resize((224, 224))

        # 이미지 컬러 스페이스 조정
        img = img.convert('RGB')

        # 이미지를 배열로 변환
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)     # 3차원으로 변환
        x = preprocess_input(x)

        # 이미지 피처 추출
        feature = self.model.predict(x)[0]

        return feature / np.linalg.norm(feature)        # 2-norm(Euclidean norm)