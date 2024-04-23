from __future__ import absolute_import
from __future__ import division
import torchvision.transforms as transforms

train_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=63 / 255),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                    )
test_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                    )

def image_augment(state:str='train'):
    if state == 'train':
        return train_transform
    else:
        return test_transform
