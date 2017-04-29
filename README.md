## Image Classifier
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project implements a classifier for the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
using the [LeNet 5 neural network architecture](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) created by Yann LeCun. 
LeNet 5 is implemented in Python using Google TensorFlow.


### Installation

Clone the repo:

    git clone https://github.com/alexhagiopol/image_classifier
    cd image_classifier

Download the [German Traffic Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) with annotations in a Python Pickle format that is easy to work with:
        
    wget https://www.dropbox.com/s/mwjjb67xbpj6rgp/traffic-signs-data.tar.gz
    tar -xvzf traffic-signs-data.tar.gz
    rm -rf traffic-signs-data.tar.gz


    