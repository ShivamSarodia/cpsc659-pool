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
    labels.append((str(4*(idx-1)+1).rjust(3, '0'), label))
    im.save(out_dir + '/' + str(4*(idx-1)+1).rjust(3, '0') + '.png')

    im = im.resize(dims)
    for i, rot in enumerate([90, 180, 270]):
        labels.append((str(4*(idx-1)+i+2).rjust(3, '0'), label))
        newim = im.rotate(rot)
        newim.save(out_dir + '/' + str(4*(idx-1)+i+2).rjust(3, '0') + '.png')

def main():
    image_dirs = ['training_data/solids/', 'training_data/stripes/', 'training_data/none/']
    for i, dir in enumerate(image_dirs):
        for idx in range(25):
            print("Augmenting " + dir+str(i*25+idx+1).rjust(3, '0') + '.png')
            augment_data(dir+str(i*25+idx+1).rjust(3, '0') + '.png', 'train_augmented', (20, 20), i*25+idx+1, i)

    print(labels)
    np.save('train_augmented_labels', np.array(labels))

if __name__ == '__main__':
    main()
