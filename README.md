# Hand Gestures detection
The purpose of this project is to perform hand gestures detection based on Deep Learning using pytorch. Two types of neural networks are used , the Feed-Forward Neural Netwroks and the Convolutional Neural Networks.

## FNN Based detection

The FNN used in this project is a fully meshed network, with 3 hidden layers of 256 ReLU activated units each.
This Networks classifies contours based on their Fourrier Descriptors

### Videos

For each classifier, we have test and trainning videos in the /data directory.

### Features extaction
To extract features from videos , we need first to extract the coordiantes of the contours, this is done by the C++ script ``` extract_coord.cpp ```.This script needs to be buit using cmake.



```bash
$ cmake .
$ make
$ ./extract_coord <videofile>
```
Here you can use the trackbar to enhance the detection, once you've decided what is the best threshold, you can rerun the script in order to get the file with the contour.

```bash
$./extract_coord <videofile> <threshold>
```

The file will be saved in the same directory and the same name as the video file.

After extracting contours from all videos it's time to use the result files in order to extract features from these files.

To do so you need to use the script ``` extract_feat_aut.py```.

```bash
$python3 extract_feat_aut.py <contour1> <contour2> ...
```
The outputs here are csv files with same name as the input file with _feat at the end, with the following format : 

<-feature1->,<-feature2->...<-feature21-> \
<-feature1->,<-feature2->...<-feature21->


### Dataset Preparation
Once features are extracted now we need to prepare a dataset from these features. The script responsible of doing this is ``` extract_feat_aut.py``` .It is used the following way : 
```bash
$python3 agg_data_aut.py <features1_csv> <label1> <features2_csv> <label2> .....
```
At the end, we will have our dataset in the file ``` all_data.csv ``` ready to use for training.

This file has the following format : 

<-label->,<-feature1->,<-feature2->...<-feature21->

### Trainning Classifier
The next step after extracting the data, is to train the classifier with is the FNN here. 
```bash
$python3 train.py --classes <nbclasses>
```
where <-nbclasses-> is the number of classes for classification
### Testing Classifier
To test this classifier 2 options are available, either to test it using the camera or using a video file .
To check how to use the test scipt : 
```bash
$python3 test.py -h
```
## CNN Based detection
The CNN used in this project is a Vanilla Network, with 2 convolutional layers, followed by 3 fully connected layers with 256 ReLU activated units.

The first convolutional layer contains 32 (3x3) convolutions with a stride of 1 followed by a ReLu anda (2x2) MaxPooling with stride = 2.

The second one is similar but with 64 convolutions.

### Generating data

In this Network, generating data is easier, it is done using a set of videos.

```bash
$python3 generate_dataset.py <vid1> <label1> <feat2> <label2> ...
```

### Training and Testing
Training and testing this network is exactly the same as the FNN but with the scripts ``` train_cnn.py ``` and ``` test_cnn.py ```.