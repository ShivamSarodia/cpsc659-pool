import os
import sys
import math
import cv2
from PIL import Image
import numpy as np

labels = []

def augment_data(img_path, out_dir, dims, idx, label):
    global labels
    im = Image.open(img_path)
    im = im.resize(dims)
    labels.append((str(4*(idx-1)+1).rjust(3, '0'), label))
    im.save(out_dir + '/' + str(4*(idx-1)+1).rjust(3, '0') + '.png')

    for i, rot in enumerate([90, 180, 270]):
        labels.append((str(4*(idx-1)+i+2).rjust(3, '0'), label))
        newim = im.rotate(rot)
        newim.save(out_dir + '/' + str(4*(idx-1)+i+2).rjust(3, '0') + '.png')

def main():
    image_dirs = ['ball_imgs/solids/', 'ball_imgs/stripes/']#, 'training_data/none/']
    image_aug_dirs = ['training_data_sorted/solids/', 'training_data_sorted/stripes/']#, 'training_data_sorted/none/']

    # file_cnt = 0
    # for i, dir in enumerate(image_dirs):
    #     for filename in os.listdir(dir):
    #         file_cnt += 1
    #         im = Image.open(dir + filename)
    #         im.save(image_aug_dirs[i] + str(file_cnt).rjust(3, '0') + '.png')

    for i, dir in enumerate(image_aug_dirs):
        for filename in os.listdir(dir):
            if not filename[-4:] == '.png':
                continue
            idx = int(filename[:-4])
            # for idx in range(25):
            # print("Augmenting " + dir+str(i*25+idx+1).rjust(3, '0') + '.png')
            print("Augmenting " + dir+str(idx).rjust(3, '0') + '.png')
            # augment_data(dir+str(i*25+idx+1).rjust(3, '0') + '.png', 'train_augmented', (20, 20), i*25+idx+1, i)
            augment_data(dir+str(idx).rjust(3, '0') + '.png', 'train_augmented', (34, 34), idx, i)

    print(labels)
    np.save('train_augmented_labels', np.array(labels))

if __name__ == '__main__':
    main()
