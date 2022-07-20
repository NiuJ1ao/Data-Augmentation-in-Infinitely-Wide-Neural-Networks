from imgaug import augmenters as iaa

def augment_images(train, rotate=(-45, 45), crop_percentage=(0.15, 0.15), iterations=1):
    for _ in range(iterations):
        seq = iaa.Sequential([
            iaa.Affine(rotate=rotate),
            iaa.Crop(percent=crop_percentage)
        ])
        aug_images = seq(images=train)
    
    return aug_images

if __name__ == "__main__":
    pass