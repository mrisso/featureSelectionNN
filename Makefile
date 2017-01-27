all:
	g++ files.cpp -g `pkg-config --cflags opencv` `pkg-config --libs opencv` -I $(CAFFE_HOME)/include -I $(CAFFE_HOME)/build/src -L $(CAFFE_HOME)/build/lib -lcaffe -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcuda -std=c++11 -lglog -lboost_system
