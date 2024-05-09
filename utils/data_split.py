import os
import shutil

def split(dataset_name='CIFAR-FS'):
    """
        将数据集划分到 train val test 三个文件夹
    """

    set = ['train', 'val', 'test']

    image_floder_path = 'D:/dataset/cifar100/CIFAR-FS-data/image'
    txt_folder_path = 'D:/dataset/cifar100/CIFAR-FS-data/splits'
    dst_floder = 'D:/dataset/cifar100/CIFAR-FS-data'

    # set: train val test
    for set_name in set:
        with open(os.path.join(txt_folder_path, set_name + '.txt')) as file:
            tmp = file.readlines()
            for label in tmp:
                if not os.path.exists(os.path.join(dst_floder, 'CIFAR-FS', set_name, label.strip())):
                    shutil.move(os.path.join(image_floder_path, label.strip()), os.path.join(dst_floder, 'CIFAR-FS', set_name))


split()
