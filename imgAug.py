#pip install mlxtend
import random
import os
import json
import sys
import datetime
from common import *
import argparse
parser = argparse.ArgumentParser()
# for input / output paths
parser.add_argument("--dataset_dir",default="dataset",help='Path of image datas')
parser.add_argument("--digits",default="[1,2,4,6,3, 5, 0]",help='digits of input')
parser.add_argument("--spacing_range",default="[5,10]",help='space range between two digit image')
parser.add_argument("--image_width", default=200,type=int,help='image width of output')
parser.add_argument("--dataout_dir",default='dataout',help='folder of images output')
parser.add_argument("--image_count",default=100,type=int,help='count of images output')

args = parser.parse_args()
def generate_numbers_sequence(digits, spacing_range, image_width):
    df_dt ,dt_gp= load_dataset(args.dataset_dir)
    img_out=generate_numbers_sequence_speedup(digits, spacing_range, image_width, df_dt, dt_gp,scaleFlg=False)
    return img_out

def generate_numbers_sequence_speedup(digits, spacing_range, image_width,df_dt ,dt_gp, scaleFlg=True):
    height=28
    width=28
    img_out=np.zeros((height,image_width))
    imglen=df_dt.shape[1]-1
    space_array,img_width=space_width_cal(digits, spacing_range, image_width)
    if img_width < 10:
        raise ValueError('please expand your image width')
    pos_start=0
    for i,digit in enumerate(digits):
        index=random.choice(dt_gp.groups[int(digit)])
        img=df_dt.iloc[index, 0:imglen].values
        img=img.reshape(height,width)
        img_transform=cv2.resize(img,(img_width,height))
        if scaleFlg:
            img_transform=random_scale_img(img_transform)
        pos_end=pos_start+img_width
        img_out[:,pos_start:pos_end] =img_transform
        pos_start=pos_end+space_array[i]

    img_out=min_max_normalization(img_out)
    return img_out

def random_scale_img(img, scale_in=0.2):
    assert scale_in > 0, "Please input a positive float"
    scale = (max(-1, -scale_in), scale_in)
    img_shape = img.shape

    scale_x = random.uniform(*scale)
    scale_y = random.uniform(*scale)
    resize_scale_x = 1 + scale_x
    resize_scale_y = 1 + scale_y
    shift_scalex = -img_shape[0]*scale_in/2
    shift_scaley = -img_shape[1]*scale_in/2
    img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)
    canvas = np.zeros(img_shape, dtype=np.uint8)
    x_lim = int(min(resize_scale_x, 1) * img_shape[1])
    y_lim = int(min(resize_scale_y, 1) * img_shape[0])
    canvas[:y_lim, :x_lim] = img[:y_lim, :x_lim]
    img = canvas
    img = np.array(img, dtype=np.float32)
    img=shift_x(img, shift_scalex)
    img=shift_y(img, shift_scaley)

    return img
def main():
   os.makedirs(args.dataout_dir,exist_ok=True)
   df_train, gp = load_dataset(args.dataset_dir)
   for i in range(int(args.image_count)):
       # img=generate_numbers_sequence(args.digits, args.spacing_range, args.image_width)
       img=generate_numbers_sequence_speedup(json.loads(args.digits), json.loads(args.spacing_range), int(args.image_width), df_train, gp)
       fn_id=datetime.datetime.now().strftime("%Y%m%d%H%S") + "_{0:02d}".format(i)
       fn_out=args.dataout_dir+"/"+fn_id+".png"
       cv2.imwrite(fn_out,img*255)
if __name__ == '__main__':
    main()