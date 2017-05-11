## Multi-Scale CNN Classifier 

#### Abstract
This project uses Google TensorFlow to implement a multi-scale convolutional neural network architecture created using concepts from [LeNet 5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (LeCun, 1998),
the [Sermanet & LeCun's multi-scale CNN architecture](https://drive.google.com/open?id=0B_huqLwo5sS1RzVxMlFKV0RrSmc) (2011), and [the dropout method](https://drive.google.com/open?id=0B_huqLwo5sS1QXd3S0NJY2pNeFk) (Srivastava, 2014). 

The classifier's performance is tested using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) on which it achieves 99.1% validation 
accuracy and 97.2% test accuracy. These results are encouraging given that human performance on this dataset is 98.8% (Sermanet & LeCun, 2011). However, overfitting the
GTSRB dataset remains a challenge when attempting to generalize to *any* image of German traffic signs captured by *any* camera. Future work
includes further research into generalization including further preprocessing to make reliable inferences on images from any input source.

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

#### Model Architecture

I implemented the multiscale architecture described in Sermanet and LeCun's paper with added regularization and dropout
as described in Vivek Yadav's [blog post](https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad).
The architecture contains three "stacks" consisting of two convolutional layers, one ReLU layer, one max pooling layer, and one dropout layer.
Stack one feeds into stack two which feeds into stack three. As described by Sermanet and LeCun, the output of stacks 1, 2, and 3
are combined into a single, flattened vector which is then connected to a fully connected layer, a dropout layer, a second
fully connected layer, and a second dropout layer in that order. Finally, reglarization is performed. The model architecture is 
summarized below:

| Layer         		| Description    	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| ReLU					|												|
| Max Pooling	      	| 2x2 stride,               outputs 16x16x32    |
| Dropout         	    | 50% likelihood      				      		|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| ReLU					|												|
| Max Pooling	      	| 2x2 stride,               outputs 8x8x64      |
| Dropout         	    | 50% likelihood      				      		|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| ReLU					|												|
| Max Pooling	      	| 2x2 stride,               outputs 4x4x128     |
| Dropout				| 50% Likelihood            outputs 4x4x128   	|
| Flatten + Concatenate | 	                        outputs 1x14336     |
| Fully Connected		|                           outputs 1x1024      |
| Dropout               |                           outputs 1x1024      |
| Fully Connected       |                           outputs 1x1024      |
| Dropout               |                           outputs 1x1024      |
| Fully Connected       |                           outputs 1x43        |
| Regularization        |                                               |  
|                       |                                               |
|                       |                                               |

#### Training
To train the model architecture above, I set up CUDA and cuDNN on my Ubuntu machine as described and
trained using an NVIDIA GTX 980Ti. I used a batch size of 128, 0.0002 learning rate, 50% 
dropout likelihood, and 10 epochs. After each epoch, I check if the accuracy achieved is the
highest ever, and I save the model if so. This aloows me to keep the best weights configuration
after each epoch. This training configuration takes about 1 hour on my GPU.

#### Approach

My highest validation accuracy is 99.1% and my test accuracy is 97.2%. These results are encouraging given that human performance is 98.8%.
My solution approach was to first implement the unmodified LeNet architecture with which I was not able to achieve above 85% accuracy. 
Next, I implemented Sermanet & LeCun's 2011 paper on my own. I improved that architecture by adding dropoout which was developed 3 years after
Sermanet and LeCun published their paper. I then searched the Internet for more optimized implementations to 
push my accuracy higher. I saw Vivek Yadav's blog post where he suggests doubling the number of convolutional layers in addition to 
adding regularization to the network. I implemented these changes and achieved 99.1% validation accuracy. The key insight from the literature on
this topic is the multi-scale convolutional approach. The idea is to create a network that learns high-abstraction, mid-abstraction,
and low-abstraction image features to perform classification. This is why the outputs of the first, second, and third convolutional 
groups are flattened and concatenated before fully-connected layers. The final result will be based on an evaluation of all levels
of feature abstraction.

#### Acquiring New Images from Outside the Dataset

I found five images of German traffic sings from outside the dataset on the Internet. I then resized these images to 32x32x3 and applied
the same preprocessing I applied to the GTSRB dataset. Results of this procedure are below:

![augmented_new_imgs](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/figures/augmented_new_imgs.png)

#### Performance on New Images

Unfortunately, the model was only able to correctly predict the identity of 3 of 5 signs, giving it a 60% accuracy rate. 
This accuracy is much lower than the accuracy on the test set of 97%. This indicates that the model has overfit the GTSRB 
dataset and does not properly generalize to German traffic signs in general. Future work includes investigating this overfitting
and attempting to alleviate the issue with perhaps better data augmentation. One issue I see with the differences between
the new images and the dataset images is that the new images represent the traffic signs with higher clarity and less blur than the
GTSRB dataset. Perhaps additional preprocessing on new images could help achieve better results.

#### Model Certainty

Below we show the softmax probabilities for each new image. The figure shows that the model is extremely overconfident: it is
100% certain of its wrong inferences. Future work includes researching how to overcome such issues with overconfidence and 
not generalizing to images from outside the source dataset.

![new_imgs_perf](https://github.com/alexhagiopol/multiscale_CNN_classifier/blob/master/figures/new_imgs_perf.png)




