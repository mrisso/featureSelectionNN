## Synopsis

This project is a feature selection tool used to classify road images using neural networks. It uses KITTI's semantics database provided by Alexander Hermans and Georgios Floros, consisting of 203 labeled images from their visual odometry dataset.
We are currently in **development** and **testing** phases and it's uses are still very limited.

## Requirements

- You should have caffe installed and the variable CAFFE_HOME set to the folder where your caffe installation is.

- You should also have opencv2.

## Installation

- Clone the repository

- Download the database from KITTI's webpage: https://omnomnom.vision.rwth-aachen.de/data/rwth_kitti_semantics_dataset.zip

- Extract the file into the repository's folder, renaming the new folder to 'semantics'. The hierarchy should look like this: 
(repository's folder)/semantics/labels_new
(repository's folder)/semantics/images

- To compile, simply type 'make' on terminal. 

## Usage

For now, you can run it by typing ./a.out with some numbers as arguments. The numbers represent which iteration step you want the network to save the output image. For example:

```
./a.out 0 5
```
In this case, the program will save the network output into "results" folder after the first(0) and sixth(5) iterations. It also saves, for future comparison, the corresponding image and expected labels, and a map showing the "wrong" pixels as white and the right ones as black.

You can also tweak the network by editing the model and solver files
