from PIL import Image
from model.preprocess import remove_background
import model.preprocess as preprocess

fe = preprocess.FeatureExtractor()

# 타깃 이미지 반환
def get_query(target_path):
    
    # 이미지 불러온 후 배경 제거
    img = Image.open(target_path).convert('RGBA')
    img = remove_background(img)
    
    # 이미지 피처 추출
    query = fe.extract(img)

    return query