#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <opencv/ml.h>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;






char ch[30];

//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
Ptr<DescriptorMatcher> Match = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extract = new SurfDescriptorExtractor();
SurfFeatureDetector detector(100);
//---dictionary size=number of cluster's centroids
int clusterPoints = 10;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 3;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(clusterPoints, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extract, Match);



// various tracking parameters (in seconds)
const double MHI_DURATION = 0.5;
const double threshold_value = 25;
Mat SAMHI_5 = Mat(480, 640, CV_32FC1, Scalar(0, 0, 0));;
int main(int argc, char** argv)
{
    VideoCapture capture = 0;
    
    cout<<"Training Started.... "<<endl;
    cout<<"Creating Motion History Image.... "<<endl;
        FILE *fp = fopen("train.txt", "r");
        char filename[1024];
        float label;
        if(!fp) {
            cerr << "Error: Could not find training index file "<< filename << endl;
        }
        while(!feof(fp)) {
            fscanf(fp, "%f %s", &label, filename);
            cout << "Processing file " << filename << endl;
            capture = VideoCapture(filename);
            Mat image_prev, imgToGray, image_abs_diff;
            int noOfFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
            for (int i = 1; i < noOfFrames; i++)
            {
                Mat frame;
                // Retrieve a single frame from the capture
                capture.read(frame);
                
                cvtColor(frame, imgToGray, CV_BGR2GRAY);
                
                
                int num = 5;
                
                if (i == 1){
                    image_prev = imgToGray.clone();
                }
                if (i % num == 0){
                    absdiff(image_prev, imgToGray, image_abs_diff);
                    image_prev = imgToGray.clone();
                }
                
                //to create an initial samhi image of 5;
                if (i == num + 1){
                    threshold(image_abs_diff, image_abs_diff, threshold_value, 255, THRESH_BINARY);
                    Size framesize = image_abs_diff.size();
                    int h = framesize.height;
                    int w = framesize.width;
                    SAMHI_5 = Mat(h, w, CV_32FC1, Scalar(0, 0, 0));
                    updateMotionHistory(image_abs_diff, SAMHI_5, (double)i / noOfFrames, MHI_DURATION);
                }
                //creating SAMHI image of 5
                if (i > num + 1 && i % num == 0){
                    threshold(image_abs_diff, image_abs_diff, threshold_value, 255, THRESH_BINARY);
                    updateMotionHistory(image_abs_diff, SAMHI_5, (double)i / noOfFrames, MHI_DURATION);
                }
                
            }
            
            cout << "Motion History Image Done" << endl;
            // mhi done
            vector<KeyPoint> keypoint;
            SAMHI_5.convertTo(SAMHI_5, CV_8UC1, 255, 0);
            detector.detect(SAMHI_5, keypoint);
            Mat features;
            extract->compute(SAMHI_5, keypoint, features);
            bowTrainer.add(features);
        }
    vector<Mat> descriptors = bowTrainer.getDescriptors();
    int count = 0;
    for (vector<Mat>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
    {
        count += iter->rows;
    }
    cout << "Vector quantization " << count << " features for finding the Training attributes for the corresponding class" << endl;
    Mat NumberOfClusters = bowTrainer.cluster();
    bowDE.setVocabulary(NumberOfClusters);
    Mat labels(0, 1, CV_32FC1);
    Mat trainingData(0, clusterPoints, CV_32FC1);
    int k = 0;
    vector<KeyPoint> keypoint1;
    Mat bowDescriptor1;
        
        FILE *fp2 = fopen("train.txt", "r");
        if(!fp2) {
            cerr << "Error: Could not find training index file "<< filename << endl;
        }
        while(!feof(fp2)) {
            fscanf(fp2, "%f %s", &label, filename);
            cout << "Processing file " << filename << endl;
            capture = VideoCapture(filename);
            Mat image_prev, imgToGray, image_abs_diff;
            int noOfFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
            for (int i = 1; i < noOfFrames; i++)
            {
                Mat frame;
                // Retrieve a single frame from the capture
                capture.read(frame);
                
                cvtColor(frame, imgToGray, CV_BGR2GRAY);
                
                
                int num = 5;
                
                if (i == 1){
                    image_prev = imgToGray.clone();
                }
                //to perform differences between frames adjacent by 5 frames.
                if (i % num == 0){
                    absdiff(image_prev, imgToGray, image_abs_diff);
                    image_prev = imgToGray.clone();
                }
                
                //to create an initial samhi image of 5
                if (i == num + 1){
                    threshold(image_abs_diff, image_abs_diff, threshold_value, 255, THRESH_BINARY);
                    Size framesize = image_abs_diff.size();
                    int h = framesize.height;
                    int w = framesize.width;
                    SAMHI_5 = Mat(h, w, CV_32FC1, Scalar(0, 0, 0));
                    updateMotionHistory(image_abs_diff, SAMHI_5, (double)i / noOfFrames, MHI_DURATION);
                }
                //creating SAMHI image of 5
                if (i > num + 1 && i % num == 0){
                    threshold(image_abs_diff, image_abs_diff, threshold_value, 255, THRESH_BINARY);
                    updateMotionHistory(image_abs_diff, SAMHI_5, (double)i / noOfFrames, MHI_DURATION);
                }
                
            }
            //extracting histogram in the form of bow for each image
            SAMHI_5.convertTo(SAMHI_5, CV_8UC1, 255, 0);
            detector.detect(SAMHI_5, keypoint1);
            
            
            bowDE.compute(SAMHI_5, keypoint1, bowDescriptor1);
            
            trainingData.push_back(bowDescriptor1);
            
            labels.push_back(label);
        }
    //Setting up SVM parameters
    CvSVMParams params;
    params.kernel_type = CvSVM::RBF;
    params.svm_type = CvSVM::C_SVC;
    params.gamma = 0.50625000000000009;
    params.C = 312.50000000000000;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
    CvSVM svm;
    
    
    
    printf("%s\n", "Creating the model for testing using SVM classifier");
    
    bool res = svm.train(trainingData, labels, cv::Mat(), cv::Mat(), params);
    
    Mat groundTruth(0, 1, CV_32FC1);
    Mat evalData(0, clusterPoints, CV_32FC1);
    k = 0;
    vector<KeyPoint> keypoint2;
    Mat bowDescriptor2;
    
    
    Mat results(0, 1, CV_32FC1);
    
    
    
    /// Testing
        FILE *fp3 = fopen("test.txt", "r");
        if(!fp3) {
            cerr << "Error: Could not find training index file "<< filename << endl;
        }
        while(!feof(fp3)) {
            fscanf(fp3, "%f %s", &label, filename);
            cout << "Processing file " << filename << endl;
            capture = VideoCapture(filename);
            Mat image_prev, imgToGray, image_abs_diff;
            int noOfFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
            for (int i = 1; i < noOfFrames; i++)
            {
                
                Mat frame;
                // Retrieve a single frame from the capture
                capture.read(frame);
                
                cvtColor(frame, imgToGray, CV_BGR2GRAY);
                
                
                int num = 5;
                
                if (i == 1){
                    image_abs_diff = imgToGray.clone();
                }
                //to perform differences between frames adjacent by 5 frames.
                if (i % num == 0){
                    absdiff(image_prev, imgToGray, image_abs_diff);
                    image_prev = imgToGray.clone();
                }
                
                //to create an initial samhi image of 5;
                if (i == num + 1){
                    threshold(image_abs_diff, image_abs_diff, threshold_value, 255, THRESH_BINARY);
                    Size framesize = image_abs_diff.size();
                    int h = framesize.height;
                    int w = framesize.width;
                    SAMHI_5 = Mat(h, w, CV_32FC1, Scalar(0, 0, 0));
                    updateMotionHistory(image_abs_diff, SAMHI_5, (double)i / noOfFrames, MHI_DURATION);
                }
                //creating SAMHI image of 10
                if (i > num + 1 && i % num == 0){
                    threshold(image_abs_diff, image_abs_diff, threshold_value, 255, THRESH_BINARY);
                    updateMotionHistory(image_abs_diff, SAMHI_5, (double)i / noOfFrames, MHI_DURATION);
                }
                
            }
            SAMHI_5.convertTo(SAMHI_5, CV_8UC1, 255, 0);
            detector.detect(SAMHI_5, keypoint2);
            bowDE.compute(SAMHI_5, keypoint2, bowDescriptor2);
            
            evalData.push_back(bowDescriptor2);
            groundTruth.push_back(label);
            
            
            float calculatedLabel = svm.predict(bowDescriptor2);
            
            cout << "reponse :: " << calculatedLabel << endl;
            results.push_back(calculatedLabel);
        }

    
    
    
    //calculate the number of unmatched classes 
    double errorRate = (double)countNonZero(groundTruth - results) / evalData.rows;
    double accuracy = (double) 1 - errorRate;
    printf("%s%f%s", "Accuracy is ", accuracy * 100 , "%");
    
    
}