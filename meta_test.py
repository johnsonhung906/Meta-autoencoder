from    torchvision.transforms import transforms
import  numpy as np
import  torchvision


def to_img(self, tensor):
    """
        transform output tensor from normalized image to original image
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    to_PIL_image = torchvision.transforms.TOPILImage()
    return to_PIL_image(inv_normalize(inv_normalize(tensor)))