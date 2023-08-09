//4 bit prime numbers neural network

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <iomanip>
#include <conio.h>

using namespace cv;
using namespace ml;
using namespace std;

void print(Mat& mat, int prec)
{
	for (int i = 0; i < mat.size().height; i++)
	{
		cout << "[";
		for (int j = 0; j < mat.size().width; j++)
		{
			cout << fixed << setw(2) << setprecision(prec) << mat.at<float>(i, j);
			if (j != mat.size().width - 1)
				cout << ",";
			else
				cout << "]";
		}
	}
}

int main()
{
	const int hiddenLayerSize = 16;
	float inputTrainingDataArray[16][4] = {
		  { 0.0, 0.0, 0.0, 0.0 },
		  { 0.0, 0.0, 0.0, 1.0 },
		  { 0.0, 0.0, 1.0, 0.0 },
		  { 0.0, 0.0, 1.0, 1.0 },
		  { 0.0, 1.0, 0.0, 0.0 },
		  { 0.0, 1.0, 0.0, 1.0 },
		  { 0.0, 1.0, 1.0, 0.0 },
		  { 0.0, 1.0, 1.0, 1.0 },
		  { 1.0, 0.0, 0.0, 0.0 },
		  { 1.0, 0.0, 0.0, 1.0 },
		  { 1.0, 0.0, 1.0, 0.0 },
		  { 1.0, 0.0, 1.0, 1.0 },
		  { 1.0, 1.0, 0.0, 0.0 },
		  { 1.0, 1.0, 0.0, 1.0 },
		  { 1.0, 1.0, 1.0, 0.0 },
		  { 1.0, 1.0, 1.0, 1.0 },
		  
	};
	Mat inputTrainingData = Mat(16, 4, CV_32F, inputTrainingDataArray);

	float outputTrainingDataArray[16][1] = {
		  { 0.0 },
		  { 0.0 },
		  { 1.0 },
		  { 1.0 },
		  { 0.0 },
		  { 1.0 },
		  { 0.0 },
		  { 1.0 },
		  { 0.0 },
		  { 0.0 },
		  { 0.0 },
		  { 1.0 },
		  { 0.0 },
		  { 1.0 },
		  { 0.0 },
		  { 0.0 }
	};
	Mat outputTrainingData = Mat(16, 1, CV_32F, outputTrainingDataArray);

	Ptr<ANN_MLP> mlp = ANN_MLP::create();

	Mat layersSize = Mat(3, 1, CV_16U);
	layersSize.row(0) = Scalar(inputTrainingData.cols);
	layersSize.row(1) = Scalar(hiddenLayerSize);
	layersSize.row(2) = Scalar(outputTrainingData.cols);
	mlp->setLayerSizes(layersSize);

	mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);

	TermCriteria termCrit = TermCriteria(
		//TermCriteria::Type::COUNT + TermCriteria::Type::EPS,100000000,0.000000000000000001);
		TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 1000000, 0.0000001);
	mlp->setTermCriteria(termCrit);
	mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
	Ptr<TrainData> trainingData = TrainData::create(inputTrainingData, SampleTypes::ROW_SAMPLE, outputTrainingData);

	printf("Training...\n");
	mlp->train(trainingData);

	for (int i = 0; i < inputTrainingData.rows; i++)
	{
		Mat sample = Mat(1, inputTrainingData.cols, CV_32F, inputTrainingDataArray[i]);
		Mat result;
		mlp->predict(sample, result);
		cout << sample << " = ";
		print(result, 9);
		cout << endl;
	}

	printf("Press anykey to exit... where's the anykey?");
	_getch();
	return 0;
}
