from model import getdata, similarity, visualization
from settings import directory, target_path
import warnings
warnings.filterwarnings('ignore')

# 전체 이미지 경로, 전체 이미지 피처, 타깃 이미지 피처 불러오기
img_paths, features = getdata.get_features(directory)
query = getdata.get_query(target_path)

# 타깃 이미지와의 유사도 계산
scores = similarity.calculate_similarity(img_paths, features, query)

# 타깃 이미지와 유사한 포켓몬 시각화
visualization.visualize(scores)