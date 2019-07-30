# Image Style Transfer

We implement **A Neural Algorithm of Artistic Style**, an algorithm to perform image style transfer, which is introduced in this [paper]. The algorithm is realized in Tensorflow.  

## Prerequisites

To run our code, you need to download a pre-trained [VGG network][VGG], which is an about 535MB [mat file][net] named **imagenet-vgg-verydeep-19.mat**, and put it on the top level of our folder. You could check the details of the network [here][VGG paper].

## Running

Import our functions into your python code. 

```
from style_transfer import*
```

Then you could apply the method by calling **Style_Transfer** function.

```
output_image = Style_Transfer(content_file, style_file, vgg_data, output_file, 
	content_layer, style_layers, coef_content, coef_style, 
	learning_rate, max_iter, checkpoint_iter, print_iter)
```

For the detailed manual, you could check the Jupyter notebook **style_transfer.ipynb**.

## Example

The following python codes show how to use the **Style_Transfer** function. 

You need to specify the image for content construction (i.e. **content_file**), the image for style construction (i.e. **style_file**), the generated image by this algorithm (i.e. **output_file**), the layer used for content representation (i.e. **content_layer**), the layers used for style representation (i.e. **style_layers**), the learning rate for [Adam][adam] optimizer (i.e. **learning_rate**), the maximum number of iterations (i.e. **max_iter**), the number of iterations between check points (i.e. **checkpoint_iter**), the number of iterations between printing out images (i.e. **print_iter**), the weights for the loss of content (i.e. **coef_content**), and the weights for the loss of style (i.e. **coef_style**).

```
content_file = 'input_image/samford_hall.jpg'
style_file = 'input_image/starry_night.jpg'  
vgg_data = 'imagenet-vgg-verydeep-19.mat'
output_file = 'output_image/content'
content_layer = 'conv1_1'
style_layers = ['conv1_1']
learning_rate = 1
max_iter, checkpoint_iter, print_iter = (500, 100, 100)
coef_content, coef_style = (1, 0)

output_image = Style_Transfer(content_file, style_file, vgg_data, output_file, 
	content_layer, style_layers, coef_content, coef_style, 
	learning_rate, max_iter, checkpoint_iter, print_iter)
```

Then the generated images will be saved in **output_image** folder. 

## Dependencies

* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
* [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
* [Timeit](https://docs.python.org/2/library/timeit.html)
* [Matplotlib](https://matplotlib.org/users/installing.html)


## Authors

**Fan Gao** 

Contributors who participated in this project: **Yang Shi & Chao Yin**.

## License

Copyright (c) 2018 Fan Gao. Released under GPLv3.

## Acknowledgments

* Professor Zoran Kostic
* TAs in the course ECBM E4040


[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[VGG paper]: https://arxiv.org/pdf/1409.1556.pdf
[net]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
[VGG]: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
[adam]: http://arxiv.org/abs/1412.6980
