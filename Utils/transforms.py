import torchvision.transforms as transforms
import Dataset.dataset as dataset

image_h = 480
image_w = 640

def transforms_hha():
    return transforms.Compose([dataset.scaleNorm_hha(),
                               dataset.RandomScale_hha((1.0, 1.4)),
                               dataset.RandomHSV((0.9, 1.1),
                                         (0.9, 1.1),
                                         (25, 25)),
                               dataset.RandomCrop_hha(image_h, image_w),
                               dataset.RandomFlip(),
                               dataset.ToTensor_hha(),
                               dataset.Normalize_hha()])