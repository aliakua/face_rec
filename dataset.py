
from torch.utils.data import Dataset, DataLoader
import glob
import PIL
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import numpy as np

# разные режимы датасета
DATA_MODES = ['train', 'valid', 'test']
# все изображения будут масштабированы к размеру 160*160 px
RESCALE_SIZE = 160


class CelebaDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, mode,  augmentations = None):
        super().__init__()
        self.mode = mode
        # список файлов для загрузки
        self.imgs_path = "/kaggle/working/celeba_500_class/"
        file_list = glob.glob(self.imgs_path + self.mode + "/" + "*")
        labels = []
        self.data = []
        self.augmentations = augmentations

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        for class_path in file_list:
            label = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([str(img_path), int(label)])
                labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def load_sample(self, file):
        image = Image.open(file) 
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Imagenet mean and std
        img_path, class_name = self.data[index]
        img = self.load_sample(img_path) 
        img = np.array(img) 
        # cropping - # обрежем и оставим только зону лица  и нормализуем
        img = Image.fromarray(img[77:-41, 50:-50]) 
        img = self._prepare_sample(img)  
        # standartization
        img = np.array(img / 255, dtype='float32') 
        # transformation
        if self.augmentations is None:
            img_tensor = data_transforms(img)  
        else:
            augmented  = self.augmentations(image = img)
            img_tensor = augmented['image']
        class_id = class_name
        return img_tensor, class_id

    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)