import matplotlib
import matplotlib.pyplot as plt
import torchvision
import numpy as np

matplotlib.use("TkAgg")


def show_image(image, title=None):
    npimg = image.numpy()

    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)


def show_images_grid(images):
    show_image(torchvision.utils.make_grid(images, padding=2))