import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from skimage.io import imread
import os

annos = pd.read_table('/kaggle/input/celeba-train-500/celebA_train_500/celebA_anno.txt', delim_whitespace=True, names=('filename', 'class'),
                   dtype={'filename': str, 'class': np.int64})
train_split = pd.read_table('/kaggle/input/celeba-train-500/celebA_train_500/celebA_train_split.txt', delim_whitespace=True, names=('filename', 'split'),
                   dtype={'filename': str, 'class': np.int64})


root = '/kaggle/input/celeba-train-500/celebA_train_500/celebA_imgs'
paths = []
names = []
for dirs, _, filenames in os.walk(root):
    for filename in filenames:
        paths += [os.path.join(root + '/' + filename)]
        names += [os.path.join(filename)]


d = {'filename': names, 'path': paths}
img_names = pd.DataFrame(data=d)
df = pd.merge(pd.merge(img_names, annos, on = 'filename'), train_split, on = 'filename')
df = df.sort_values(by='split').reset_index(drop=True)

table = pd.pivot_table(df, values=['filename'], index=['class','split'],
                       aggfunc={'filename': "count"})
table= table.reset_index(drop=False)
table.columns = ['class','split','cnt']

df_cond = pd.merge(df, table , how = 'outer', on = ['class','split'] )
df = df_cond.loc[df_cond['cnt'] >1]
df = df.sort_values(by='split').reset_index(drop=True)

os.system('mkdir /kaggle/working/celeba_500_class')
os.system('mkdir /kaggle/working/celeba_500_class/train')
os.system('mkdir /kaggle/working/celeba_500_class/valid')
os.system('mkdir /kaggle/working/celeba_500_class/test')