import gzip
import numpy as np
import pandas as pd
import cv2
def min_max_normalization(x):
    x=np.array(x, dtype=np.float)
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result
def load_img(file_name,dataset_dir):
    file_path = dataset_dir + '/' + file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)

    return data
def load_label(file_name,dataset_dir):
    file_path = dataset_dir + '/' + file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels
def load_dataset(dataset_dir):
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }
    dataset = {}
    dataset['train_img'] = load_img(key_file['train_img'],dataset_dir)
    dataset['train_label'] = load_label(key_file['train_label'],dataset_dir)
    dataset['test_img'] = load_img(key_file['test_img'],dataset_dir)
    dataset['test_label'] = load_label(key_file['test_label'],dataset_dir)
    df_train = pd.DataFrame(dataset['train_img'])

    df_train["label"] = dataset['train_label']
    gp =df_train.groupby('label')
    return df_train,gp
def space_width_cal(digits, spacing_range, image_width):
    space_lst=[]
    img_width_all=image_width
    cnt=len(digits)
    for i in range(cnt-1):
        space_i=np.random.randint(spacing_range[0],spacing_range[1])
        space_lst.append(space_i)
        img_width_all=img_width_all-space_i
    space_lst.append(0)
    img_width=int(img_width_all/cnt)
    space_array=np.array(space_lst)
    return space_array,img_width

def shift_x(image, shift):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += shift
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))
def shift_y(image, shift):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += shift # shift pixels
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))