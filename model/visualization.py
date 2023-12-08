import matplotlib.pyplot as plt
from PIL import Image

def visualize(scores):
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

    print('Completed Successfully!')