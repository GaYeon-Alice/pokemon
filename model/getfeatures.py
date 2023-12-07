import os
from PIL import Image
import model.preprocess as preprocess
from model.preprocess import remove_background

fe = preprocess.FeatureExtractor()

# 전체 이미지의 경로와 피처 반환
def get_features(directory):

    features = []
    img_paths = []

    img_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]

    for img_name in img_files:
        
        try:
            # 이미지 경로 저장
            image_path = os.path.join(directory, img_name)
            img_paths.append(image_path)

            # 이미지 불러온 후 배경 제거
            img = Image.open(image_path).convert('RGBA')
            img = remove_background(img)

            # 이미지 피처 추출
            feature = fe.extract(img)
            features.append(feature)

        except Exception as e:
            print('예외가 발생했습니다.', e)
    
    return img_paths, features