import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentasi untuk training
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    
    # Rotasi kecil untuk variasi pose
    A.Rotate(limit=10, p=0.5),
    
    # Perubahan brightness & contrast kecil
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    
    # Sedikit zoom/crop random
    A.RandomResizedCrop(
        height=224,
        width=224,
        scale=(0.9, 1.1),
        p=0.5
    ),
    
    # Blur ringan untuk simulasi kamera buram
    A.GaussianBlur(blur_limit=3, p=0.2),
    
    # Normalisasi sesuai ImageNet
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])

# Transformasi untuk validation/testing
test_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])
