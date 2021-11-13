import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity


def prepare_image_for_imshow(image):
    # turn channel first into channel last
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1,2,0))

    # rescale image outrange
    if image.dtype != np.uint8:
        image = rescale_intensity(image, out_range=np.uint8).astype(np.uint8)

    return image

##################################################
# Displaying image and mask pair 
##################################################
def display_pair(image1, image2, title1='Image', title2='Mask'):
    image1 = prepare_image_for_imshow(image1)

    fig = plt.figure(figsize=(16,9))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(image1)
    ax1.axis('off')
    ax1.set_title(title1)

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(image2)
    ax2.axis('off')
    ax2.set_title(title2)

    plt.show()
