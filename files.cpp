#define OK																	0
#define NULL_VECTOR															1
#define ERROR_LOADING_IMAGE													2

#define NUM_IMAGES_TO_TRAIN													10
#define CLASS_MULTIPLIER													25

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <signal.h>
#include <ctype.h>

#define USE_OPENCV

#include <caffe/caffe.hpp>
#include <caffe/solver.hpp>
#include <caffe/solver_factory.hpp>
#include <caffe/sgd_solvers.hpp>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/sigmoid_cross_entropy_loss_layer.hpp>
#include <caffe/layers/euclidean_loss_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>

using namespace std;
using namespace cv;

const string IMAGE_TRAIN_FILE = "./train.txt";
const string IMAGE_TEST_FILE= "./test.txt";

const string IMAGES_PATH = "./semantics/images/";
const string TARGET_PATH = "./semantics/labels_new/";

int requested_to_exit = 0;

void
shutdown(int sign)
{
	if (sign == SIGINT)
	{
		printf("Exit requested\n");
		requested_to_exit = 1;
	}
}

vector<string> readPathFromFile(string tFile)
{
	vector<string> files; 
	string line;

	ifstream file(tFile.c_str());

	if(file.is_open())
	{
		while(getline(file,line))
			files.push_back(line);
		file.close();
	}
	
	return files;
}

void printVectors(vector<string> v)
{
	int size = v.size();
	for(int i = 0; i < size; i++)
		cout << v[i] << "\n";
}

Mat readImage(string imageFileName, string dir)
{
	Mat image;
	Mat imageResized(Size(300,100),CV_8UC3);

	image = imread(dir + imageFileName,CV_LOAD_IMAGE_COLOR);
	if(!image.data)
	{
		cout << "Could not open image \n";
		exit(ERROR_LOADING_IMAGE);
	}

	resize(image,imageResized,imageResized.size());
	return imageResized;
}

int getClass(vector<pair<Scalar,int>> &v, Scalar color)
{
	for(int i = 0; i<v.size(); i++)
	{
		if(v[i].first == color)
			return v[i].second;
	}
	v.push_back(pair<Scalar,int>(color,v.size()));
	return v.size() - 1;
}

Mat convertImageToClasses(Mat image)
{
	unsigned char b,g,r;
	Mat classifiedImage(image.size(),CV_8UC1);
	vector<pair<Scalar,int>> classColorRelationship;
	for(int i = 0; i < image.rows; i++)
	{
		for(int j = 0; j < image.cols; j++)
		{
			b = image.data[3 * (i * image.cols + j)];
			g = image.data[3 * (i * image.cols + j) + 1];
			r = image.data[3 * (i * image.cols + j) + 2];

			Scalar color(b,g,r);
			
			classifiedImage.data[i * image.cols + j] = getClass(classColorRelationship,color);
		}
	}

	return classifiedImage;
}

int main(void)
{
	vector<string> imagesTrain;
	vector<string> imagesTest;

	if((imagesTrain = readPathFromFile(IMAGE_TRAIN_FILE)).size() == 0)
	{
		cout << "Images Vector is empty\n";
		return NULL_VECTOR;
	}

	if((imagesTest = readPathFromFile(IMAGE_TEST_FILE)).size() == 0)
	{
		cout << "Images Test Vector is empty\n";
		return NULL_VECTOR;
	}

	signal(SIGINT, shutdown);

	boost::shared_ptr<caffe::Solver<float>> solver;
	boost::shared_ptr<caffe::Net<float>> net;

	caffe::SolverParameter solver_param;

	caffe::ReadProtoFromTextFileOrDie("solver.prototxt", &solver_param);
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	net = solver->net();

	boost::shared_ptr<caffe::MemoryDataLayer<float>> input = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("input"));
	boost::shared_ptr<caffe::MemoryDataLayer<float>> target = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("target"));

	vector<int> dummyLabels;
	dummyLabels.push_back(0);

	while(!requested_to_exit)
	{
		double totalLoss = 0;
		for(int i = 0; i < NUM_IMAGES_TO_TRAIN; i++)
		{
			int p = rand() % imagesTrain.size();
			Mat image = readImage(imagesTrain[p],IMAGES_PATH);
			Mat targetImage = readImage(imagesTrain[p],TARGET_PATH);

			vector<Mat> v;
			v.push_back(image);
			
			input->AddMatVector(v,dummyLabels);

			vector<Mat> vTarget;
			vTarget.push_back(convertImageToClasses(targetImage));

			target->AddMatVector(vTarget,dummyLabels);

			solver->Step(2);
			totalLoss += net->blob_by_name("loss")->cpu_data()[0];
		}
		printf("total loss: %lf\n", totalLoss);

		for(int i = 0; i < imagesTest.size(); i++)
		{

		}
	}

//	namedWindow("Image", WINDOW_AUTOSIZE);
//	imshow("Image", imageResized);
//
//	namedWindow("Expected", WINDOW_AUTOSIZE);
//	imshow("Expected",expectedImageResized);
//	char key = waitKey(0);
//
//	if(key == 'p') i -= 1;
//	if (key == 'n') i += 1;
//	if (i < 0) i = size - 1;
//	if (i >= size) i = 0;
//
	return OK;
}
