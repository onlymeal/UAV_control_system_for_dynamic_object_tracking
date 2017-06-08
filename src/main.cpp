#include "ardrone/ardrone.h"
#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2\nonfree\features2d.hpp>

#define M_PI 3.14159265358979323846

#define MOVE_FORWARD	1
#define MOVE_BACKWARD	2
#define TURN_LEFT		3
#define TURN_RIGHT		4
#define MOVE_UP			5
#define MOVE_DOWN		6

using namespace std;
using namespace cv;

static const int NUMBER_OF_SCALE_STEPS = 3;
static const int NUMBER_OF_ROTATION_STEPS = 18;
static const float SMALLEST_SCALE_CHANGE = 0.5;
static const int FRAME_WIDTH = 640;
static const int FRAME_HEIGHT = 360;

// AR.Drone class
ARDrone ardrone;

enum APPLiCATION_MODE
{
	DO_NOTHING,
	DETECTION,
	TAKING_NEW_TAMPLATE,
	END
} appMode;


int noOfPointsPickedOnTemplateImage;

bool hasAnyTemplateChosenBefore = false;

IplImage* templateImageRGBFull;
IplImage* templateImageRGB = NULL;
IplImage* templateImageGray = NULL;

IplImage* newFrameRGB;
IplImage* newFrameGray;
IplImage* sideBySideImage;

IplImage* emptyImage;
IplImage* outputImage;

CvPoint templateCorners[2];

CvMat templateObjectCorners[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

CvFont font;

int templateROIX;
int templateROIY;

float obj_pos_x;
float obj_pos_y;
float obj_size;
bool obj_detect;

int fastThreshold;

double* templateObjectCornersData[NUMBER_OF_SCALE_STEPS][NUMBER_OF_SCALE_STEPS];

//영상처리 변수
Mat select_obj;
Mat trans;
Mat inputimg;

int minHessian = 1000;

Mat des_object;

SurfFeatureDetector detector( minHessian );
SurfDescriptorExtractor extractor;
FlannBasedMatcher matcher;

inline float degreeToRadian(const float d)
{
  return (d / 180.0) * M_PI;
}

void mouseHandler(int event, int x, int y, int flags, void* params)
{
  if (appMode == TAKING_NEW_TAMPLATE) {
    templateCorners[1] = cvPoint(x, y);
    switch (event) {
    case CV_EVENT_LBUTTONDOWN:
      templateCorners[noOfPointsPickedOnTemplateImage++] = cvPoint(x, y);
      break;
    case CV_EVENT_RBUTTONDOWN:
      break;
    case CV_EVENT_MOUSEMOVE:
      if (noOfPointsPickedOnTemplateImage == 1)
	templateCorners[1] = cvPoint(x, y);
      break;
    }
  }
}

void move3d(int com){
	double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
	if (com == MOVE_FORWARD)	{ vx =  1.0;	printf("move forward\n"); }
    if (com == MOVE_BACKWARD)	{ vx = -1.0;		printf("move backward\n"); }
    if (com == TURN_LEFT)		{ vr =  1.0;	printf("turn left\n"); }
    if (com == TURN_RIGHT)		{ vr = -1.0;		printf("turn right\n"); }
    if (com == MOVE_UP)			{ vz =  1.0;	printf("move up\n"); }
    if (com == MOVE_DOWN)		{ vz = -1.0;		printf("move down\n"); }
    ardrone.move3D(vx, vy, vz, vr);
}

void waitKeyAndHandleKeyboardInput(int timeout){
	const char key = cvWaitKey(timeout);
	switch (key) {
	case 'q' : case 'Q':
		appMode = END;
		break;
	case 't': case 'T':
		appMode = TAKING_NEW_TAMPLATE;
		break;
	case 'd' : case 'D':
		appMode = DO_NOTHING;
		break;
	case ' ' :
		if (ardrone.onGround()) { ardrone.takeoff(); printf("Take off \n"); }
		else                    { ardrone.landing(); printf("Tkae down \n"); }
		break;
	case 'i' :
		move3d(MOVE_FORWARD);
		break;
	case 'k' :
		move3d(MOVE_BACKWARD);
		break;
	case 'j' :
		move3d(TURN_LEFT);
		break;
	case 'l' :
		move3d(TURN_RIGHT);
		break;
	case ';' :
		move3d(MOVE_UP);
		break;
	case '.':
		move3d(MOVE_DOWN);
		break;
	}
}

void saveCornersCoors(void)
{
	const double templateWidth = templateImageGray->width;
	const double templateHeight = templateImageGray->height;

	double* corners = templateObjectCornersData[0][0];
	corners[0] = 0;
	corners[1] = 0;
	corners[2] = templateWidth;
	corners[3] = 0;
	corners[4] = templateWidth;
	corners[5] = templateHeight;
	corners[6] = 0;
	corners[7] = templateHeight;
}

bool saveNewTemplate(void)
{
	const int templateWidth = templateCorners[1].x - templateCorners[0].x;
	const int templateHeight = templateCorners[1].x - templateCorners[0].x;

	templateROIX = templateCorners[0].x, templateROIY = templateCorners[0].y;

	const CvSize templateSize = cvSize(templateWidth, templateHeight);
	const CvRect templateRect = cvRect(templateCorners[0].x, templateCorners[0].y, templateWidth, templateHeight);

	cvCopy(newFrameRGB, templateImageRGBFull);

	cvReleaseImage(&templateImageRGB);
	templateImageRGB = cvCreateImage(templateSize, IPL_DEPTH_8U, 3);

	cvReleaseImage(&templateImageGray);
	templateImageGray = cvCreateImage(templateSize, IPL_DEPTH_8U, 1);

	cvSetImageROI(newFrameGray, templateRect);
	cvCopy(newFrameGray, templateImageGray);
	cvResetImageROI(newFrameGray);

	cvSetImageROI(newFrameRGB, templateRect);
	cvCopy(newFrameRGB, templateImageRGB);
	cvResetImageROI(newFrameRGB);

	saveCornersCoors();
	
	return true;
}

void putImagesSideBySide(IplImage* result, const IplImage* img1, const IplImage* img2)
{
  const int bigWS = result->widthStep;
  const int bigHalfWS = result->widthStep >> 1;
  const int lWS = img1->widthStep;

  char *p_big = result->imageData;
  char *p_bigMiddle = result->imageData + bigHalfWS;
  const char *p_l = img1->imageData;
  for (int i = 0; i < FRAME_HEIGHT; ++i, p_big += bigWS, p_bigMiddle += bigWS) {
    memcpy(p_big, p_l + i*lWS, lWS);
  }
}

void takeNewTemplateImage(void)
{
	cvCopyImage(newFrameRGB, outputImage);
	cv::Rect rect(templateCorners[0],templateCorners[1]);
	switch (noOfPointsPickedOnTemplateImage) {
	case 1:
//		cvRectangle(outputImage, templateCorners[0], templateCorners[1], cvScalar(0, 255, 0), 0);
		select_obj = trans(rect);
		break;
	case 2:
		if (saveNewTemplate()) {
		appMode = DETECTION;
		hasAnyTemplateChosenBefore = true;
		}
		noOfPointsPickedOnTemplateImage = 0;
		break;
	 default:
		break;
	}
}
bool takeNewFrame(void)
{
	if ((newFrameRGB = ardrone.getImage())){
		cvCvtColor(newFrameRGB, newFrameGray, CV_BGR2GRAY);}
	else
		return false;
	return true;
}

void showOutput(IplImage* img)
{
  static char text[256];

  if (appMode != TAKING_NEW_TAMPLATE) {
    sprintf(text, "2013. 10. 29 ByungJoon Kwon");
    cvPutText(img, text, cvPoint(10, 450), &font, cvScalar(255, 255, 255));
  }
  
  cvShowImage("ARDrone_Tracking_TEST", img);
}

//먼저 서프부터
void doDetection(void){
	
	double t=0;
	t=(double)getTickCount();
	Mat object = select_obj;
	
	std::vector<Point2f> obj_corners(4);

	obj_corners[0] = cvPoint(0,0);
	obj_corners[1] = cvPoint( object.cols, 0 );
	obj_corners[2] = cvPoint( object.cols, object.rows );
	obj_corners[3] = cvPoint( 0, object.rows );
	
	Mat des_image, img_matches;
	std::vector<KeyPoint> kp_object;
    std::vector<KeyPoint> kp_image;
    std::vector<vector<DMatch>> matches; 
    std::vector<DMatch > good_matches;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    std::vector<Point2f> scene_corners(4);
    Mat H;
    Mat image;
        
    cvtColor(inputimg, image, CV_RGB2GRAY);

	detector.detect( object, kp_object );  //요기 에러남!!!
	extractor.compute( object, kp_object, des_object );

    detector.detect( image, kp_image );
    extractor.compute( image, kp_image, des_image );

    matcher.knnMatch(des_object, des_image, matches, 2);
	
    for(int i = 0; i < min(des_image.rows-1,(int) matches.size()); i++) 
    {
        if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
        {
            good_matches.push_back(matches[i][0]);
        }
    }
		
    drawMatches( object, kp_object, image, kp_image, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		
    if (good_matches.size() >= 4)
    {
        for( int i = 0; i < good_matches.size(); i++ )
        {
            obj.push_back( kp_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
        }

        H = findHomography( obj, scene, CV_RANSAC );

        perspectiveTransform( obj_corners, scene_corners, H);

        line( img_matches, scene_corners[0] + Point2f( object.cols, 0), scene_corners[1] + Point2f( object.cols, 0), Scalar(255, 2, 0), 2 );
        line( img_matches, scene_corners[1] + Point2f( object.cols, 0), scene_corners[2] + Point2f( object.cols, 0), Scalar(255, 2, 0), 2 );
        line( img_matches, scene_corners[2] + Point2f( object.cols, 0), scene_corners[3] + Point2f( object.cols, 0), Scalar(255, 2, 0), 2 );
        line( img_matches, scene_corners[3] + Point2f( object.cols, 0), scene_corners[0] + Point2f( object.cols, 0), Scalar(255, 2, 0), 2 );
    }
		
    imshow( "Matches", img_matches );
	t=1/(((double)getTickCount() - t) / getTickFrequency());
	printf("%lffps  ",t);
	float pos_x = (floor)(scene_corners[0].x+scene_corners[1].x+scene_corners[2].x+scene_corners[3].x)/4;
	float pos_y = (floor)(scene_corners[0].y+scene_corners[1].y+scene_corners[2].y+scene_corners[3].y)/4;
	float size = (((scene_corners[1].x-scene_corners[0].x)+(scene_corners[2].x-scene_corners[3].x))/2)*((((scene_corners[3].y-scene_corners[0].y)+(scene_corners[2].y-scene_corners[1].y))/2));
	if(good_matches.size()>= 4){
		printf("position : x = %g, y = %g	",pos_x,pos_y);
		printf("area : %g \n", size);
		if(obj_detect == false){
			obj_pos_x = pos_x;
			obj_pos_y = pos_y;
			obj_size = size;
		}
		else{
			if(obj_pos_x -20.0 > pos_x) move3d(TURN_LEFT);
			else if(obj_pos_x +20.0 < pos_x) move3d(TURN_RIGHT);
			//if(obj_pos_y -20.0 > pos_y) move3d(MOVE_UP);
			//else if(obj_pos_y +20.0 < pos_y) move3d(MOVE_DOWN);
			if(obj_size -1200.0 > size) move3d(MOVE_FORWARD);
			else if(obj_size +1200.0 < size) move3d(MOVE_BACKWARD);
		}
		obj_detect = true;
	}
	else
		printf("not detect\n");
}

void init_pos(void){
	obj_pos_x = 0;
	obj_pos_y = 0;
	obj_size = 0;
	obj_detect = false;
}

void run(void)
{
	vector<IplImage*> images;
	while (true) {
		IplImage* result = outputImage;
		switch(appMode) {
		case TAKING_NEW_TAMPLATE:
			trans = cv::cvarrToMat(templateImageRGBFull);
			init_pos();
			takeNewTemplateImage();
			putImagesSideBySide(sideBySideImage, outputImage, emptyImage);
			result = sideBySideImage;
			break;
		case DETECTION:
			inputimg = cvarrToMat(newFrameRGB);
			takeNewFrame();
			cvCopyImage(newFrameRGB, outputImage);
			putImagesSideBySide(sideBySideImage, newFrameRGB, templateImageRGBFull);
			doDetection();
			result = sideBySideImage;
			break;
		case DO_NOTHING:
			takeNewFrame();
			outputImage=newFrameRGB;
			putImagesSideBySide(sideBySideImage, newFrameRGB, emptyImage);
			result = sideBySideImage;
			break;
		case END:
			return ;
		default:
			break;
		}
		showOutput(result);

		waitKeyAndHandleKeyboardInput(10);
	}
}

void init(void)
{
	srand(time(NULL));
	appMode = DO_NOTHING;	

	newFrameGray = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 1);
	outputImage = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
	templateImageRGBFull = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
	sideBySideImage = cvCreateImage(cvSize( FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
	emptyImage = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
	templateImageRGB = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 3);
	templateImageGray = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 1);
	for (int s = 0; s < NUMBER_OF_SCALE_STEPS; s++) {
		for (int r = 0; r < NUMBER_OF_ROTATION_STEPS; r++) {
		  templateObjectCornersData[s][r] = new double[8];
		  templateObjectCorners[s][r] = cvMat(1, 4, CV_64FC2, templateObjectCornersData[s][r]);
		}
	}

	cvNamedWindow("ARDrone_Tracking_TEST", CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback("ARDrone_Tracking_TEST", mouseHandler, NULL);

	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN | CV_FONT_ITALIC, 1, 1, 0, 1);

}

int main(int argc, char **argv)
{
    

    // Initialize
    if (!ardrone.open()) {
        printf("Failed to initialize.\n");
        return -1;
    }

    // Battery
    printf("Battery = %d%%\n", ardrone.getBatteryPercentage());

    // Instructions
    printf("***************************************\n");
    printf("*   A.R Drone Object Tracking system  *\n");
    printf("*           - How to Play -           *\n");
    printf("***************************************\n");
	printf("*                                     *\n");
	printf("* - Tracking -                        *\n");
    printf("*    't'     -- Capture enw tamplate  *\n");
//	printf("*    'd'     -- en/disable tracking   *\n");
	printf("*                                     *\n");
    printf("* - Controls -                        *\n");
    printf("*    'Space' -- Takeoff/Landing       *\n");
    printf("*    'Up'    -- Move forward          *\n");
    printf("*    'Down'  -- Move backward         *\n");
    printf("*    'Left'  -- Turn left             *\n");
    printf("*    'Right' -- Turn right            *\n");
    printf("*    'Q'     -- Move upward           *\n");
    printf("*    'A'     -- Move downward         *\n");
    printf("*                                     *\n");
    printf("*    'Q'     -- Exit                  *\n");
    printf("*                                     *\n");
    printf("***************************************\n\n");
	
	init();
	run();
	/*
    while (1) {
        // Key input
        int key = cvWaitKey(33);
        if (key == 0x1b) break;

        // Update
        if (!ardrone.update()) break;
	
		takeNewFrame();
        // Get an image
        IplImage *image = newFrameRGB;

        // Take off / Landing 
        if (key == ' ') {
            if (ardrone.onGround()) ardrone.takeoff();
            else                    ardrone.landing();
        }

        // Move
        double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
        if (key == 0x260000) vx =  1.0;
        if (key == 0x280000) vx = -1.0;
        if (key == 0x250000) vr =  1.0;
        if (key == 0x270000) vr = -1.0;
        if (key == ';')      vz =  1.0;
        if (key == '.')      vz = -1.0;
        ardrone.move3D(vx, vy, vz, vr);

        // Change camera
        static int mode = 0;
        if (key == 'c') ardrone.setCamera(++mode%4);

        // Display the image
        cvShowImage("camera", image);
    }
	*/
    // See you
    ardrone.close();

    return 0;
}