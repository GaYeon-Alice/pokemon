import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import getquery, getfeatures
from settings import directory, target_path
import warnings
warnings.filterwarnings('ignore')

# 전체 이미지 경로, 전체 이미지 피처, 타깃 이미지 피처 불러오기
img_paths, features = getfeatures.get_features(directory)
query = getquery.get_query(target_path)

# 타깃 이미지와의 유사도 계산
dists = np.linalg.norm(features - query, axis=1)
ids = np.argsort(dists)[:30]    # 타깃 이미지를 포함하여 유사한 순으로 30개의 이미지 저장
scores = [(dists[id], img_paths[id]) for id in ids]

# 타깃 이미지와 유사한 포켓몬 시각화
axes = []
fig = plt.figure(figsize=(8,8))
for a in range(5 * 6):
    score = scores[a]
    axes.append(fig.add_subplot(5, 6, a + 1))
    subplot_title = str(score[0])
    axes[-1].set_title(subplot_title)
    plt.axis('off')
    plt.imshow(Image.open(score[1]))
fig.tight_layout()
plt.show()
