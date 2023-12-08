import numpy as np

def calculate_similarity(img_paths, features, query):

    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)[:30]    # 타깃 이미지를 포함하여 유사한 순으로 30개의 이미지 저장
    scores = [(dists[id], img_paths[id]) for id in ids]

    return scores