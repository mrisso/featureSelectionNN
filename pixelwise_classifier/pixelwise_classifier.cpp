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

#define OK 0
#define NULL_VECTOR 1
#define ERROR_LOADING_IMAGE 2

#define DEBUG 0
#define TRAINING_DEBUG 0

using namespace std;
using namespace cv;

const string IMAGE_TRAIN_FILE = "../train.txt";
const string IMAGE_TEST_FILE= "../test.txt";

const string IMAGES_PATH = "../semantics/images/";
const string TARGET_PATH = "../semantics/labels_new/";
const string RESULTS_PATH = "../results/";
const string RESULTS_PATH_DEBUG = "../results/debug/";

const int NUM_CLASSES = 17;
const int AREA_SIZE = 80;
const int NUM_TRAINS_BY_IMAGE = 100;
const int NUM_IMAGES_PER_TRAIN_SESSION = 10;

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


vector<string> 
readPathFromFile(string tFile)
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


void 
printVectors(vector<string> v)
{
	int size = v.size();
	for(int i = 0; i < size; i++)
		cout << v[i] << "\n";
}


Mat 
imageSubsampling(Mat image)
{
	Mat sampledImage(Size(204,123),CV_8UC3);

	for(int i = 0; i < sampledImage.rows; i++)
	{
		for(int j = 0; j < sampledImage.cols; j++)
		{
			int imageI = 3 * i;
			int imageJ = j * 6;

			if(imageI > image.rows || imageJ > image.cols)
				printf("Maior! i: %i j: %i ii: %i ij: %i ir: %i ic: %i\n", i, j, imageI, imageJ, image.rows, image.cols);

			int p = 3 * (imageI * image.cols + imageJ);
			int q = 3 * (i * sampledImage.cols + j);

			sampledImage.data[q + 0] = image.data[p + 0];
			sampledImage.data[q + 1] = image.data[p + 1];
			sampledImage.data[q + 2] = image.data[p + 2];
		}
	}

	return sampledImage;
}


Mat 
correctImage(Mat &image)
{
	Mat corr(Size(1226, 370), image.type());

	for (int i = 0; i < 370; i++)
		for (int j = 0; j < 1226; j++)
			for (int c = 0; c < image.channels(); c++)
				corr.data[corr.channels() * (i * corr.cols + j) + c] = image.data[image.channels() * (i * image.cols + j) + c];

	return corr;
}


Mat 
readImage(string imageFileName, string dir, bool subsampling = false)
{
	Mat image;

	image = imread(dir + imageFileName);
	if(!image.data)
	{
		cout << "Could not open image \n";
		exit(ERROR_LOADING_IMAGE);
	}

	if (image.rows != 370)
		image = correctImage(image);

	Mat sampledImage;

	if (subsampling)
		sampledImage = imageSubsampling(image);
	else
		sampledImage = image;

	if(DEBUG)
	{
		imshow("Sampled Image",sampledImage);
		imshow("Image",image);
		waitKey(0);
	}

	return sampledImage;
}


int 
getClass(vector<pair<Scalar,int>> &v, Scalar color)
{
	for(unsigned int i = 0; i<v.size(); i++)
	{
		if(v[i].first == color)
			return v[i].second;
	}
	v.push_back(pair<Scalar,int>(color,v.size()));
	return v.size() - 1;
}


Mat 
convertImageToClasses(Mat image, vector<pair<Scalar,int>> &classColorRelationship)
{
	unsigned char b,g,r;
	Mat classifiedImage(image.size(),CV_8UC1);
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


Mat 
convertClassesToImage(const float *classifiedImage,int rows, int cols, vector<pair<Scalar,int>> &classColorRelationship)
{
	Mat image(rows,cols,CV_8UC3);
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			float pd = classifiedImage[i * image.cols + j];
			int pos;

			if(pd > (classColorRelationship.size() - 1))
				pos = classColorRelationship.size() - 1;

			else if(pd < 0)
				pos = 0;

			else if(pd > (floor(pd) + 0.5))
				pos = floor(pd) + 1;

			else
				pos = floor(pd);

			Scalar color = classColorRelationship[pos].first;

			image.data[3 * (i * image.cols + j) + 0] = color[0];
			image.data[3 * (i * image.cols + j) + 1] = color[1];
			image.data[3 * (i * image.cols + j) + 2] = color[2];
		}
	}
	return image;
}


Mat 
convertClassesToImage(Mat classifiedImage, int rows, int cols, vector<pair<Scalar,int>> &classColorRelationship)
{
	Mat image(rows,cols,CV_8UC3);

	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			float pd = (float) classifiedImage.data[i * image.cols + j];
			int pos;

			if(pd > (classColorRelationship.size() - 1))
				pos = classColorRelationship.size() - 1;
			else if(pd < 0)
				pos = 0;
			else if(pd > (floor(pd) + 0.5))
				pos = floor(pd) + 1;
			else
				pos = floor(pd);

			Scalar color = classColorRelationship[pos].first;

			image.data[3 * (i * image.cols + j) + 0] = color[0];
			image.data[3 * (i * image.cols + j) + 1] = color[1];
			image.data[3 * (i * image.cols + j) + 2] = color[2];
		}
	}

	return image;
}


Mat 
imgCompare(Mat image, Mat expectedImage)
{
	Mat compImage(image.size(),CV_8UC3);
	unsigned char b,g,r;

	for(int i = 0; i < image.rows; i++)
		for(int j = 0; j < image.cols; j++)
		{
			b = image.data[3 * (i * image.cols + j) + 0];
			g = image.data[3 * (i * image.cols + j) + 1];
			r = image.data[3 * (i * image.cols + j) + 2];

			Scalar imageColor(b,g,r);
			
			b = expectedImage.data[3 * (i * expectedImage.cols + j) + 0];
			g = expectedImage.data[3 * (i * expectedImage.cols + j) + 1];
			r = expectedImage.data[3 * (i * expectedImage.cols + j) + 2];

			Scalar expectedColor(b,g,r);

			if(expectedColor == imageColor)
			{
				compImage.data[3 * (i * compImage.cols + j) + 0] = 0;
				compImage.data[3 * (i * compImage.cols + j) + 1] = 0;
				compImage.data[3 * (i * compImage.cols + j) + 2] = 0;
			}

			else
			{
				compImage.data[3 * (i * compImage.cols + j) + 0] = 255;
				compImage.data[3 * (i * compImage.cols + j) + 1] = 255;
				compImage.data[3 * (i * compImage.cols + j) + 2] = 255;
			}
		}
	return compImage;
}


bool 
showImage(vector<int> v, int number)
{
	for(unsigned int i = 0; i < v.size(); i++)
		if(v[i] == number)
			return true;
	return false;
}


int
sample_class(const float *v, int n)
{
#if 0
	int i;
	vector<double> ps(n, 0.0);
	double sum = 0.0;

	for (i = 0; i < n; i++)
	{
		ps[i] = exp(v[i]);
		sum += ps[i];
	}

	for (i = 0; i < n; i++)
	{
		ps[i] /= sum;

		if (i > 0)
			ps[i] += ps[i - 1];
	}

	// DEBUG
//	printf("V:\t");
//	for (i = 0; i < n; i++)
//		printf("%.2lf\t", v[i]);
//	printf("\nP:\t");
//	for (i = 0; i < n; i++)
//		printf("%.2lf\t", ps[i]);
//	printf("\n");

	double unary_prob = (double) rand() / (double) RAND_MAX;

	for (i = 0; i < n; i++)
		if (unary_prob < ps[i])
			return i;

	// it shouldn't happen
	return rand() % n;
#else

	int m = 0;

	for (int i = 1; i < n; i++)
		if (v[i] > v[m])
			m = i;

	return m;

#endif

}


int
main(int argc, char **argv)
{
	vector<int> numbers;

	if(argc > 1)
		for(int count = 1; count < argc; count++)
			numbers.push_back(atoi(argv[count]));

	vector<pair<Scalar,int>> classColorRelationship;
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

	caffe::ReadProtoFromTextFileOrDie("pixelwise_solver.prototxt", &solver_param);
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	net = solver->net();

	boost::shared_ptr<caffe::MemoryDataLayer<float>> input = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("input"));
	boost::shared_ptr<caffe::MemoryDataLayer<float>> target = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("target"));

	vector<int> compressionParams;
	compressionParams.push_back(CV_IMWRITE_JPEG_QUALITY);
	compressionParams.push_back(100);

	printf("imagesTrain.size(): %ld\n", imagesTrain.size());
	namedWindow("Image", WINDOW_AUTOSIZE);

	while(!requested_to_exit)
	{
		double totalLoss = 0;
		double accuracy = 0;
		int num_samples = 0;

		for(int i = 0; i < NUM_IMAGES_PER_TRAIN_SESSION; i++)
		{
			int p = rand() % imagesTrain.size();
			Mat image = readImage(imagesTrain[p],IMAGES_PATH);
			Mat targetImage = convertImageToClasses(readImage(imagesTrain[p], TARGET_PATH), classColorRelationship);

			for (int k = 0; k < NUM_TRAINS_BY_IMAGE; k++)
			{
				Rect r(rand() % (image.cols - AREA_SIZE), // x
					rand() % (image.rows - AREA_SIZE), // y
					AREA_SIZE, AREA_SIZE); // width, height

				// center of the selected area
				int px = r.x + r.width / 2;
				int py = r.y + r.height / 2;

				Mat area(image, r);
				int label = (int) (targetImage.data[py * targetImage.cols + px]);

				vector<Mat> v;
				vector<int> labels;

				v.push_back(area);
				labels.push_back(label);
		
				input->AddMatVector(v, labels);

				net->Forward();
				int pred = sample_class(net->blob_by_name("fc2")->cpu_data(), 17);

				if (pred == label) accuracy++;
				num_samples++;

				//printf("Prediction: %d Expected: %d\n", pred, label);				

				if (k % 5 == 0)
				{
					Mat view = image.clone();
					rectangle(view, r, Scalar(0,0,255), 2);
					imshow("Image", view);
					waitKey(1);
				}

				solver->Step(5);
				totalLoss += net->blob_by_name("loss")->cpu_data()[0];
			}
		}

		accuracy /= (double) num_samples;
		printf("Training total loss: %f Training accuracy: %lf\n", totalLoss, accuracy);

		accuracy = 0;
		num_samples = 0;

		for(int i = 0; i < 1; i++)
		{
			//printf("Test image %d of %ld\n", i, imagesTest.size());

			int random_id = rand() % imagesTest.size();
			Mat image = readImage(imagesTest[random_id],IMAGES_PATH);
			Mat base_target = readImage(imagesTest[random_id], TARGET_PATH);
			Mat targetImage = convertImageToClasses(base_target, classColorRelationship);
			Mat miniTarget = Mat::zeros(image.rows / 4, image.cols / 8, CV_8UC1);
			Mat estimatedClasses = Mat::zeros(image.rows / 4, image.cols / 8, CV_8UC1);

			for (int row = 0, a = 0; row < (image.rows - AREA_SIZE); row += 4, a++)
			{
				//printf("Testing row %d of %d\n", row, image.rows - AREA_SIZE);

				for (int col = 0, b = 0; col < (image.cols - AREA_SIZE); col += 8, b++)
				{
					Rect r(col, row, AREA_SIZE, AREA_SIZE);

					int px = r.x + r.width / 2;
					int py = r.y + r.height / 2;

					Mat area(image, r);

					if ((row * image.cols + col) % 100 == 0)
					{
						Mat view = image.clone();
						rectangle(view, r, Scalar(0,0,255), 2);
						imshow("Image", view);
						waitKey(1);
					}

					vector<Mat> v;
					vector<int> labels;

					v.push_back(area);
					labels.push_back(0);
		
					input->AddMatVector(v, labels);

					net->Forward();

					int pred = sample_class(net->blob_by_name("fc2")->cpu_data(), 17);
					int label = (int) (targetImage.data[py * targetImage.cols + px]);

					if (pred == label) accuracy++;
					num_samples++;

					estimatedClasses.data[a * estimatedClasses.cols + b] = (unsigned char) pred;
					miniTarget.data[a * miniTarget.cols + b] = label;
				}
			}

			Mat outputImage = convertClassesToImage(estimatedClasses, estimatedClasses.rows, estimatedClasses.cols, classColorRelationship);
			Mat outputTarget = convertClassesToImage(miniTarget, miniTarget.rows, miniTarget.cols, classColorRelationship); 
			Mat compImage = imgCompare(outputImage, outputTarget);

			imshow("Image", image);
			imshow("Base", base_target);
			imshow("Target", outputTarget);
			imshow("Output", outputImage);
			imshow("Comp", compImage);

			waitKey(1);
		}

		accuracy /= (double) num_samples;
		printf("Test accuracy: %lf\n", accuracy);

	}

	return OK;
}


