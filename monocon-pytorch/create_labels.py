import os
import shutil

paths = os.listdir('../data/training/image_2')

for path in paths:
    if path.split('.')[0][-4:] == '_fog':
        org_label_path = '../data/training/label_2/' + path.split('.')[0][:-4] + '.txt'
        new_label_path = '../data/training/label_2/' + path.split('.')[0] + '.txt'
        shutil.copyfile(org_label_path, new_label_path)