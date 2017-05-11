### Multi-Scale CNN Classifier 
This project uses Google TensorFlow to implement a multi-scale convolutional neural network architecture created using concepts from [LeNet 5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (LeCun, 1998),
the [Sermanet & LeCun's multi-scale CNN architecture](https://drive.google.com/open?id=0B_huqLwo5sS1RzVxMlFKV0RrSmc) (2011), and [the dropout concept](https://drive.google.com/open?id=0B_huqLwo5sS1QXd3S0NJY2pNeFk) (Srivastava, 2014). 
We test the classifier's performance using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

#### Installation
This procedure was tested on Ubuntu 16.04 and Mac OS X 10.11.6 (El Capitan). GPU-accelerated training is supported on Ubuntu only.

Prerequisites: Install Python package dependencies using [my instructions.](https://github.com/alexhagiopol/deep_learning_packages) Then, activate the environment:

    source activate deep-learning

Optional, but recommended on Ubuntu: Install support for NVIDIA GPU acceleration with CUDA v8.0 and cuDNN v5.1:

    wget https://www.dropbox.com/s/08ufs95pw94gu37/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    wget https://www.dropbox.com/s/9uah11bwtsx5fwl/cudnn-8.0-linux-x64-v5.1.tgz
    tar -xvzf cudnn-8.0-linux-x64-v5.1.tgz
    cd cuda/lib64
    export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
    cd ..
    export CUDA_HOME=`pwd`
    sudo apt-get install libcupti-dev
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl

Clone the image_classifier repo:

    git clone https://github.com/alexhagiopol/image_classifier
    cd image_classifier

Download the [German Traffic Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) with annotations in a Python Pickle format that is easy to work with:
        
    wget https://www.dropbox.com/s/mwjjb67xbpj6rgp/traffic-signs-data.tar.gz
    tar -xvzf traffic-signs-data.tar.gz
    rm -rf traffic-signs-data.tar.gz

#### Execution
Perform preprocessing and data augmentation:

    python preproc.py
    
Run the code to train and validate the model on your machine:

    python main.py

### Technical Report
The implementation and reluts can be viewed simultaneously in the [Traffic_Sign_CLassifier.ipynb
iPython notebook](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/Traffic_Sign_Classifier.ipynb)
   

#### Dataset Summary
The GTSRB dataset contains 51839 total images, each annotated with one of 43 sign classes. Each image is a cropped traffic sign
from frames in a vehicle dashcam. The dataset contains images that often blurry, too dark, or captured from
challenging view angles. For this project, the dataset is divided into 34799 training examples, 4410 validation 
examples, and 12630 testing examples. A sample of raw images in the dataset is shown below:

![raw_images](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/figures/raw.png)

#### Exploratory Visualization

This graphic shows the class distributions of the training examples and 
validation examples in the input dataset. It's clear that the training and
validation sets have different class distributions i.e. some classes may be overrepresented
in one set but underrepresented in another set. This observation motivates data augmentation.

![raw_dist_train](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/figures/raw_dist_train.png)

![raw_dist_valid](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/figures/raw_dist_valid.png)

#### Preprocessing

I followed the guidance of Sermanet and LeCun's 2011 paper linked above and began by grayscaling all input images. The authors
state that grayscaling yielded higher accuracy results in their experiments. A possible reason is that sign color may be a
misleading indicator of sign class: several sign types share the same color, and the appearance of color may be diminished 
under poor lighting conditions such as those we frequently observe in the raw data. Next, I used the contrast limited
adaptive histogram equalization algorithm (CLAHE) to equalize the histograms in the input images. This has the effect of
making underexposed or overexposed images contain pixel values that are numerically closer to a proper exposure. It is intended to
mitigate the effects of poor lighting in the input dataset. My final preprocessing step was to normalize the images such that
their pixel value range is from -1 to 1 instead of 0 to 255. This is intended to ensure numerical stability during the 
weight value optimization procedure. 

The next preprocessing step was data augmentation. It was clear that the class distribution of the training set was not
the same as that of the validation set. I saw this as an opportunity for a model trained on the training set to fail to 
make correct inferences on the validation set. Furthermore, Sermanet and LeCun encourage data augmentation to push accuracy higher.
They recommend perturbing the input images with random rotation, translation, shear, scale, brightness, and blur effects to generate "new"
labeled training examples. I implemented these augmentation strategies, and I provide the figures below to show examples of the
augmented dataset and the class distribution of the augmented dataset. 

![augmented](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/figures/augmented.png)

![augmented_dist_train](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/figures/augmented_dist_train.png)

Unfortunately, after over 60 hours of training and experimenting using an NVIDIA GTX 980Ti, I was not able to achieve higher 
accuracy with augmented data than with unagumented data. This result baffles me; I am stuck at 94% accuracy at the present moment.
The provided implementations forgo augmentation which yielded an accuracy of 93%.

#### Model Architecture

I implemented the multiscale architecture described in Sermanet and LeCun's paper with added regularization and dropout
as described in Vivek Yadav's [blog post](https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad).
The architecture contains three "stacks" consisting of two convolutional layers, one ReLU layer, one max pooling layer, and one dropout layer.
Stack one feeds into stack two which feeds into stack three. As described by Sermanet and LeCun, the output of stacks 1, 2, and 3
are combined into a single, flattened vector which is then connected to a fully connected layer, a dropout layer, a second
fully connected layer, and a second dropout layer in that order. Finally, reglarization is performed. The model architecture is 
summarized graphically below:

| Layer         		| Description    	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| ReLU					|												|
| Max Pooling	      	| 2x2 stride,               outputs 16x16x32    |
| Dropout         	    | 75% likelihood      				      		|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| ReLU					|												|
| Max Pooling	      	| 2x2 stride,               outputs 8x8x64      |
| Dropout         	    | 75% likelihood      				      		|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| ReLU					|												|
| Max Pooling	      	| 2x2 stride,               outputs 4x4x128     |
| Dropout				| 75% Likelihood            outputs 4x4x128   	|
| Flatten + Concatenate | 	                        outputs 1x14336     |
| Fully Connected		|                           outputs 1x1024      |
| Dropout               |                           outputs 1x1024      |
| Fully Connected       |                           outputs 1x1024      |
| Dropout               |                           outputs 1x1024      |
| Fully Connected       |                           outputs 1x43        |
|                       |                                               |  
|                       |                                               |
|:---------------------:|:---------------------------------------------:|

#### Training

#### Solution Approach

#### Acquiring New Images from Outside the Dataset

#### Performance on New Images

#### Model Certainty


