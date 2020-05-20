# Coding Project: MNIST digits sequence generator
The goal of this project is to write a program that can generate images
representing sequences of numbers, for data augmentation purposes.
These images would be used to train classifiers and generative deep learning
models.  A script that saves examples of generated images is helpful in
inspecting the characteristics of the generated images and in inspecting the
trained models behaviours.
 
## Specifications
* 0.digit dataset:[MNIST database](http://yann.lecun.com/exdb/mnist/), 
* 1.generate an image of a sequence
* 2.the digits have to be stacked horizontally
* 3.the spacing between them should follow a uniform distribution over a range
determined by two user specified numbers. Each digit in the generated sequence is
then chosen randomly from one of its representations in the MNIST dataset.
* 4.The width of the output image in pixels is specified by the user
* 5.the height should be 28 pixels 
* 6.function:generate_numbers_sequence(digits, spacing_range, image_width):
* The generated image is saved in the current directory as a .png.

## Note that besides implementing the expected function all, you are free to
implement as you wish. If you have time, you may also want to expand the
project and think about additional methods for data expansion (warping, etc.).
```
* add the function of scale transformation 
* wait for expansion: rotation , shear, perspective and so on.
```
## What should be provided

* a python module (or package) that defines the main function. 
* That package should be written such as it can be imported by a 3rd party  program.
  implementation details.
* you can import the function:[from imgAug import generate_numbers_sequence,random_scale_img]
* We expect the code to be organized, designed, tested and documented as if it
were going into production.
```
excute result is outputed into the local folder:[dataout]'
```
* You can send the work assignement as a zipfile, tarball, etc. or better yet the
whole code repository (git, hg, etc.). Just make sure it is not put in a public
place !
```
zip , git
```
## Scoring
* We value quality over feature-completeness. It is fine to leave things aside
provided you call them out in your project's README. Part of the goal of this
exercise is to help us identify what you consider production-ready code.

## assess  code:

* clarity: is the code organized in well defined functions, with separated
  concerns ? Is it implemented in a way that makes it difficult or simple to
  extend ? Depending on your advertised python expertise, we will also look
  whether the code is idiomatic python.
* documentation: does the README clearly and concisely explains the problem and solution ?
  Are technical tradeoffs explained ?
* testing and correctness: the code needs to do what is asked, and you need to be able to
  explain why it is correct. Ideally, the code would have some tests. We're not
  looking for full coverage (given time constraint) but just trying to get a
  feel for your testing skills.
* technical choices: do choices of libraries, architecture etc. seem appropriate for the
  chosen application?
* going the last mile: if you find the problem trivial, can you think of more
  advanced data augmentation techniques for the problem of handwritten digit
  recognition ? Can you think of a way to generate many images
  
```
dvanced data augmentation techniques
1.rotation  
2.shear
3.perspective 
4.and so on.
```

```
Can you think of a way to generate many images
the following function will output many images by user defination
:args.image_count
def main():
   os.makedirs(args.dataout_dir,exist_ok=True)
   df_train, gp = load_dataset(args.dataset_dir)
   for i in range(args.image_count):
       # img=generate_numbers_sequence(args.digits, args.spacing_range, args.image_width)
       img=generate_numbers_sequence_speedup(args.digits, args.spacing_range, args.image_width, df_train, gp)
       fn_id=datetime.datetime.now().strftime("%Y%m%d%H%S") + "_{0:02d}".format(i)
       fn_out=args.dataout_dir+"/"+fn_id+".png"
       cv2.imwrite(fn_out,img*255)
```

# excute program
## digit dataset
```
 download data from [MNIST database](http://yann.lecun.com/exdb/mnist/)
 save the data to [dataset]
```
##  package install 
```
install the following package with pip or conda
$ opencv-python            4.1.1.26
$ pandas                   0.25.1
$ numpy                    1.16.4
```
##  function excute and explain 
```
1.  generate_numbers_sequence(digits, spacing_range, image_width)
    generate one image of numbers sequence
    digits:[3, 5, 0]
    spacing_range:(10,30)
    image_width:100
2.  generate_numbers_sequence_speedup(digits, spacing_range, image_width,df_dt ,dt_gp, scaleFlg=True):
    digits:[3, 5, 0]
    spacing_range:(10,30)
    image_width:100
    scaleFlg:define the scale change or not
    df_dt ,dt_gp: prepare data by call function[load_dataset]
 * before excute this function , please firstly excute the following code,so when you produce multiple image,
 the speed will be up since the input data loading and preprocess is excute only one time.
  df_dt ,dt_gp = load_dataset(args.dataset_dir)
 * scaleFlg is used to define the scale transformation, if the true is defined, the scale transform will be 
   excuted , and the data augment become more strong
3.  random_scale_img(img, scale_in=0.2)
  scale transform function
  img:input image
  scale_in: define the scale, for example 0.2 is defined ,the output is random transform at the range [-0.2,0.2]
  for the axis of x , y
```

# exctute sample
```
python imgAug.py \
--dataset_dir="dataset" \
--digits="[1,2,3]" \
--spacing_range="[5,10]" \
--image_width=200 \
--dataout_dir="testout50" \
--image_count=50
```
