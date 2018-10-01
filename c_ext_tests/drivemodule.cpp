#include <Python.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/shm.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#define image_width 640
#define image_height 480
#define resize_width 280
#define resize_height 210
#define semantic_width 320
#define semantic_height 660
#define INDICATORS_SIZE 16

struct shared_use_st
{  
    int written;  //a label, if 1: available to read, if 0: available to write
    uint8_t data[image_width*image_height*3];  // image data field  
    int control;
    int pause;
    double fast;

    double dist_L;
    double dist_R;

    double toMarking_L;
    double toMarking_M;
    double toMarking_R;

    double dist_LL;
    double dist_MM;
    double dist_RR;

    double toMarking_LL;
    double toMarking_ML;
    double toMarking_MR;
    double toMarking_RR;

    double toMiddle;
    double angle;
    double speed;

    double steerCmd;
    double accelCmd;
    double brakeCmd;
};

struct shared_use_st *shared;
void *shm = NULL;
int shmid;
IplImage* screenRGB;//=cvCreateImage(cvSize(image_width,image_height),IPL_DEPTH_8U,3);
IplImage* resizeRGB;//=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
IplImage* semanticRGB;//=cvCreateImage(cvSize(semantic_width,semantic_height),IPL_DEPTH_8U,3);
IplImage* error_bar;//=cvCreateImage(cvSize(640,180),IPL_DEPTH_8U,3);
IplImage* legend;//=cvLoadImage("Legend6.png"); // was "../torcs/Legend6.png"
IplImage* background;//=cvLoadImage("semantic_background_3lane.png"); // was "../torcs/semantic_background_3lane.png"

static int testSharedMemory() {
    // void *shm = NULL; 
    // int shmid;
 
    shmid = shmget((key_t)4567, sizeof(struct shared_use_st), 0666|IPC_CREAT);  
    if(shmid == -1)  
    {  
        fprintf(stderr, "shmget failed\n");  
        exit(EXIT_FAILURE);  
    }  

    shm = shmat(shmid, 0, 0);  
    if(shm == (void*)-1)  
    {  
        fprintf(stderr, "shmat failed\n");  
        exit(EXIT_FAILURE);  
    }  
    printf("\n********** Memory sharing started, attached at %p **********\n", shm); 

    shared = (struct shared_use_st*)shm;
    shared->written = 0;
    shared->control = 0;
    shared->pause = 0;
    shared->fast = 0.0;

    shared->dist_L = 0.0;
    shared->dist_R = 0.0;

    shared->toMarking_L = 0.0;
    shared->toMarking_M = 0.0;
    shared->toMarking_R = 0.0;

    shared->dist_LL = 0.0;
    shared->dist_MM = 0.0;
    shared->dist_RR = 0.0;

    shared->toMarking_LL = 0.0;
    shared->toMarking_ML = 0.0;
    shared->toMarking_MR = 0.0;
    shared->toMarking_RR = 0.0;

    shared->toMiddle = 0.0;
    shared->angle = 0.0;
    shared->speed = 0.0;

    shared->steerCmd = 0.0;
    shared->accelCmd = 0.0;
    shared->brakeCmd = 0.0;  

    char buffer[50];
    while (true) {
        printf("shared->written is %d\n", shared->written);

        scanf("%s", buffer);
        if (!strcmp(buffer, "x")) { // x: shutdown
            //shared->pause = 0;
            printf("Shutting down\n");
            break;
        }
        else if (!strcmp(buffer, "w")) { // w: write
            shared->written = !shared->written;
        }
    }

    ////////////////////// clean up memory sharing
    if(shmdt(shm) == -1) {  
        fprintf(stderr, "shmdt failed\n");  
        exit(EXIT_FAILURE);  
    }  

    if(shmctl(shmid, IPC_RMID, 0) == -1) {
        fprintf(stderr, "shmctl(IPC_RMID) failed\n");  
        exit(EXIT_FAILURE);  
    }
    printf("\n********** Memory sharing stopped. Good Bye! **********\n");    
    exit(EXIT_SUCCESS); 
}

static int setupSharedMemory() {
    shmid = shmget((key_t)4567, sizeof(struct shared_use_st), 0666|IPC_CREAT);  
    if(shmid == -1) {  
        fprintf(stderr, "shmget failed\n");
        return EXIT_FAILURE;
    }  

    shm = shmat(shmid, 0, 0);  
    if(shm == (void*)-1) {  
        fprintf(stderr, "shmat failed\n");
        return EXIT_FAILURE; 
    }  
    
    printf("\n********** Memory sharing started, attached at %p **********\n", shm);

    shared = (struct shared_use_st*)shm;
    shared->written = 0;
    shared->control = 0;
    shared->pause = 0;
    shared->fast = 0.0;

    shared->dist_L = 0.0;
    shared->dist_R = 0.0;

    shared->toMarking_L = 0.0;
    shared->toMarking_M = 0.0;
    shared->toMarking_R = 0.0;

    shared->dist_LL = 0.0;
    shared->dist_MM = 0.0;
    shared->dist_RR = 0.0;

    shared->toMarking_LL = 0.0;
    shared->toMarking_ML = 0.0;
    shared->toMarking_MR = 0.0;
    shared->toMarking_RR = 0.0;

    shared->toMiddle = 0.0;
    shared->angle = 0.0;
    shared->speed = 0.0;

    shared->steerCmd = 0.0;
    shared->accelCmd = 0.0;
    shared->brakeCmd = 0.0;

    return EXIT_SUCCESS;
}

static int closeSharedMemory() {
    if(shmdt(shm) == -1) {  
        fprintf(stderr, "shmdt failed\n");  
        return EXIT_FAILURE;
    }
    if(shmctl(shmid, IPC_RMID, 0) == -1) {
        fprintf(stderr, "shmctl(IPC_RMID) failed\n");  
        return EXIT_FAILURE;
    }
    printf("\n********** Memory sharing stopped. Good Bye! **********\n");    
    return EXIT_SUCCESS;
}

static bool isWritten() {
    return shared->written;
}

static void writeShared(bool written) {
    shared->written = written;
}

// The pause flag means it is paused on 0 (falsy value), going against intuition
static bool isPaused() {
    return !shared->pause;
}

// The pause flag means it is paused on 0 (falsy value), going against intuition
static void pauseDrive(bool pause) {
    shared->pause = !pause;
}

static bool isControlling() {
    return shared->control;
}

static void setControl(bool control) {
    shared->control = control;
}

static void setupOpencv() {
    screenRGB=cvCreateImage(cvSize(image_width,image_height),IPL_DEPTH_8U,3);
    resizeRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
    // semanticRGB=cvCreateImage(cvSize(semantic_width,semantic_height),IPL_DEPTH_8U,3);
    // error_bar=cvCreateImage(cvSize(640,180),IPL_DEPTH_8U,3);
    // legend=cvLoadImage("Legend6.png"); // was "../torcs/Legend6.png"
    // background=cvLoadImage("semantic_background_3lane.png"); // was "../torcs/semantic_background_3lane.png"
    cvNamedWindow("Semantic Visualization",1);
    cvNamedWindow("Image from leveldb",1);
    cvNamedWindow("Error Bar",1);
    printf("\n********** Opencv set up **********\n");
}

static uint8_t * readImage() {
    for (int h = 0; h < image_height; h++) {
       for (int w = 0; w < image_width; w++) {
            // HxWxC (BGR) image, just inverting H dimension
            screenRGB->imageData[(h*image_width+w)*3+2]=shared->data[((image_height-h-1)*image_width+w)*3+0];
            screenRGB->imageData[(h*image_width+w)*3+1]=shared->data[((image_height-h-1)*image_width+w)*3+1];
            screenRGB->imageData[(h*image_width+w)*3+0]=shared->data[((image_height-h-1)*image_width+w)*3+2];
       }
    }
    cvResize(screenRGB,resizeRGB);
    // Show image
    // cvShowImage("Image from leveldb", resizeRGB);
    // cvWaitKey(99);
    return (uint8_t *) resizeRGB->imageData;
}

static double * readIndicators() {
    double * indicators = (double *) calloc(INDICATORS_SIZE, sizeof(double));
    indicators[0]  = shared->fast;
    indicators[1]  = shared->dist_L;
    indicators[2]  = shared->dist_R;
    indicators[3]  = shared->toMarking_L;
    indicators[4]  = shared->toMarking_M;
    indicators[5]  = shared->toMarking_R;
    indicators[6]  = shared->dist_LL;
    indicators[7]  = shared->dist_MM;
    indicators[8]  = shared->dist_RR;
    indicators[9]  = shared->toMarking_LL;
    indicators[10] = shared->toMarking_ML;
    indicators[11] = shared->toMarking_MR;
    indicators[12] = shared->toMarking_RR;
    indicators[13] = shared->toMiddle;
    indicators[14] = shared->angle;
    indicators[15] = shared->speed;
    return indicators;
}

////////////////////// control parameters
float road_width=8.0;
float center_line;
float coe_steer=1.0;
int lane_change=0;
float pre_ML;
float pre_MR;
float desired_speed;
float steering_record[5]={0,0,0,0,0};
int steering_head=0;
float slow_down=100;
float dist_LL_record=30;
float dist_RR_record=30;

int left_clear=0;
int right_clear=0;
int left_timer=0;
int right_timer=0;
int timer_set=60;

float pre_dist_L=60;
float pre_dist_R=60;
float steer_trend;
////////////////////// END control parameters

////////////////////// visualization parameters
int marking_head=1;
int marking_st;
int marking_end;
int pace;
int car_pos;
int goto_lane=0;
int true_goto_lane=0;

float p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y;
CvPoint* pt = new CvPoint[4]; 
int visualize_angle=1;

float err_angle;
float err_toMarking_ML;
float err_toMarking_MR;
float err_toMarking_M;

int manual=0;
int counter=0;
////////////////////// END visualization parameters

static void controller(double *indicators) {
    double angle=indicators[14];

    double toMarking_L=indicators[3];
    double toMarking_M=indicators[4];
    double toMarking_R=indicators[5];

    double dist_L=indicators[1];
    double dist_R=indicators[2];

    double toMarking_LL=indicators[9];
    double toMarking_ML=indicators[10];
    double toMarking_MR=indicators[11];
    double toMarking_RR=indicators[12];

    double dist_LL=indicators[6];
    double dist_MM=indicators[7];
    double dist_RR=indicators[8];

    int fast;
    if (indicators[0]>0.5) fast=1;
    else fast=0;

    if (goto_lane==2 && toMarking_LL<-8) toMarking_LL=-7.5;  // correct error output
    if (goto_lane==1 && toMarking_RR>8) toMarking_RR=7.5;  // correct error output

    //////////////////////////////////////////////// a controller processes the cnn output and get the optimal steering, acceleration/brake
    if (shared->control==1) {

        slow_down=100; 

        if (pre_dist_L<20 && dist_LL<20) {   // left lane is occupied or not
            left_clear=0;
            left_timer=0;
        } else left_timer++;                

        if (pre_dist_R<20 && dist_RR<20) {   // right lane is occupied or not
            right_clear=0;
            right_timer=0;
        } else right_timer++;

        pre_dist_L=dist_LL;
        pre_dist_R=dist_RR;

        if (left_timer>timer_set) {  // left lane is clear
            left_timer=timer_set;
            left_clear=1;
        }

        if (right_timer>timer_set) {  // right lane is clear
            right_timer=timer_set;
            right_clear=1;
        }


        if (lane_change==0 && dist_MM<15) {  // if current lane is occupied

            steer_trend=steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4];  // am I turning or not

            if (toMarking_LL>-8 && left_clear==1 && steer_trend>=0 && steer_trend<0.2) {  // move to left lane
                lane_change=-2;
                coe_steer=6;
                right_clear=0;
                right_timer=0;
                left_clear=0;
                left_timer=30;
                timer_set=60;
            }

            else if (toMarking_RR<8 && right_clear==1 && steer_trend<=0 && steer_trend>-0.2) {  // move to right lane
                lane_change=2;
                coe_steer=6;
                left_clear=0;
                left_timer=0;
                right_clear=0;
                right_timer=30;
                timer_set=60;
            }

            else {
                float v_max=20;
                float c=2.772;
                float d=-0.693;
                slow_down=v_max*(1-exp(-c/v_max*dist_MM-d));  // optimal velocity car-following model
                if (slow_down<0) slow_down=0;
            }
        }

        ///////////////////////////////////////////////// prefer to stay in the central lane
        else if (lane_change==0 && dist_MM>=15) {

            steer_trend=steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4];  // am I turning or not

            if (toMarking_RR>8 && left_clear==1 && steer_trend>=0 && steer_trend<0.2) {  // in right lane, move to central lane
                lane_change=-2;
                coe_steer=6;
                left_clear=0;
                left_timer=30;
            }

            else if (toMarking_LL<-8 && right_clear==1 && steer_trend<=0 && steer_trend>-0.2) {  // in left lane, move to central lane
                lane_change=2;
                coe_steer=6;
                right_clear=0;
                right_timer=30;
            }
        }
        ///////////////////////////////////////////////// END prefer to stay in the central lane

        ///////////////////////////////////////////////// implement lane changing or car-following
        if (lane_change==0) {
            if (-toMarking_ML+toMarking_MR<5.5) {
                coe_steer=1.5;
                center_line=(toMarking_ML+toMarking_MR)/2;
                pre_ML=toMarking_ML;
                pre_MR=toMarking_MR;
                if (toMarking_M<1)
                coe_steer=0.4;
            } else {
                if (-pre_ML>pre_MR)
                center_line=(toMarking_L+toMarking_M)/2;
                else
                center_line=(toMarking_R+toMarking_M)/2;
                coe_steer=0.3;
            }
        }

        else if (lane_change==-2) {
            if (-toMarking_ML+toMarking_MR<5.5) {
                center_line=(toMarking_LL+toMarking_ML)/2;
                if (toMarking_L>-5 && toMarking_M<1.5)
                center_line=(center_line+(toMarking_L+toMarking_M)/2)/2;
            } else {
                center_line=(toMarking_L+toMarking_M)/2;
                coe_steer=20;
                lane_change=-1;
            }
        }

        else if (lane_change==-1) {
            if (toMarking_L>-5 && toMarking_M<1.5) {
                center_line=(toMarking_L+toMarking_M)/2;
                if (-toMarking_ML+toMarking_MR<5.5)
                center_line=(center_line+(toMarking_ML+toMarking_MR)/2)/2;
            } else {
                center_line=(toMarking_ML+toMarking_MR)/2;
                lane_change=0;
            }
        }

        else if (lane_change==2) {
            if (-toMarking_ML+toMarking_MR<5.5) {
                center_line=(toMarking_RR+toMarking_MR)/2;
                if (toMarking_R<5 && toMarking_M<1.5)
                center_line=(center_line+(toMarking_R+toMarking_M)/2)/2;
            } else {
                center_line=(toMarking_R+toMarking_M)/2;
                coe_steer=20;
                lane_change=1;
            }
        }

        else if (lane_change==1) {
            if (toMarking_R<5 && toMarking_M<1.5) {
                center_line=(toMarking_R+toMarking_M)/2;
                if (-toMarking_ML+toMarking_MR<5.5)
                center_line=(center_line+(toMarking_ML+toMarking_MR)/2)/2;
            } else {
                center_line=(toMarking_ML+toMarking_MR)/2;
                lane_change=0;
            }
        }
        ///////////////////////////////////////////////// END implement lane changing or car-following

        shared->steerCmd = (angle - center_line/road_width) / 0.541052/coe_steer;  // steering control, "shared->steerCmd" [-1,1] is the value sent back to TORCS

        if (lane_change==0 && coe_steer>1 && shared->steerCmd>0.1)   // reshape the steering control curve
            shared->steerCmd=shared->steerCmd*(2.5*shared->steerCmd+0.75);

        steering_record[steering_head]=shared->steerCmd;  // update previous steering record
        steering_head++;
        if (steering_head==5) steering_head=0;


        if (fast==1) desired_speed=20;
        else desired_speed=20-fabs(steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4])*4.5;
        if (desired_speed<10) desired_speed=10;

        if (slow_down<desired_speed) desired_speed=slow_down;

        ///////////////////////////// speed control           
        if (desired_speed>=shared->speed) {
            shared->accelCmd = 0.2*(desired_speed-shared->speed+1);
            if (shared->accelCmd>1) shared->accelCmd=1.0;
            shared->brakeCmd = 0.0;
        } else {
            shared->brakeCmd = 0.1*(shared->speed-desired_speed);
            if (shared->brakeCmd>1) shared->brakeCmd=1.0;
            shared->accelCmd = 0.0;
        }
        ///////////////////////////// END speed control

    }

    printf("M_LL:%.2lf, M_ML:%.2lf, M_MR:%.2lf, M_RR:%.2lf, d_LL:%.2lf, d_MM:%.2lf, d_RR:%.2lf\n", toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR, dist_LL, dist_MM, dist_RR);
    printf("M_L:%.2lf, M_M:%.2lf, M_R:%.2lf, d_L:%.2lf, d_R:%.2lf, angle:%.3lf, fast:%d\n", toMarking_L, toMarking_M, toMarking_R, dist_L, dist_R, angle, fast);
    printf("coe_steer:%.1lf, lane_change:%d, steer:%.2lf, d_speed:%d, speed:%d, l_clear:%d, r_clear:%d, timer_set:%d\n\n", coe_steer, lane_change, shared->steerCmd, int(desired_speed*3.6), int(shared->speed*3.6), left_clear, right_clear, timer_set);
    fflush(stdout);
    //////////////////////////////////////////////// END a controller processes the cnn output and get the optimal steering, acceleration/brake
}

/////////////////////
// Python Wrappers //
/////////////////////

static PyObject * testSharedMemory_py(PyObject *self, PyObject *args) {
    int i = testSharedMemory();
    return Py_BuildValue("i", i);
}

static PyObject * setupSharedMemory_py(PyObject *self, PyObject *args) {
    int err = setupSharedMemory();
    return Py_BuildValue("i", err);
}

static PyObject * closeSharedMemory_py(PyObject *self, PyObject *args) {
    int err = closeSharedMemory();
    return Py_BuildValue("i", err);
}

static PyObject * isWritten_py(PyObject *self, PyObject *args) {
    if(isWritten()) {
        // return Py_True;
        Py_RETURN_TRUE;
    }
    // return Py_False;
    Py_RETURN_FALSE;
}

static PyObject * write_py(PyObject *self, PyObject *args) {
    bool written;
    if (!PyArg_ParseTuple(args, "p", &written)) return NULL;
    writeShared(written);
    //return Py_None;
    Py_RETURN_NONE;
}

static PyObject * isPaused_py(PyObject *self, PyObject *args) {
    if(isPaused()) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject * pause_py(PyObject *self, PyObject *args) {
    bool pause;
    if (!PyArg_ParseTuple(args, "p", &pause)) return NULL;
    pauseDrive(pause);
    Py_RETURN_NONE;
}

static PyObject * isControlling_py(PyObject *self, PyObject *args) {
    if(isControlling()) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject * setControl_py(PyObject *self, PyObject *args) {
    bool control;
    if (!PyArg_ParseTuple(args, "p", &control)) return NULL;
    setControl(control);
    Py_RETURN_NONE;
}

static PyObject * setupOpencv_py(PyObject *self, PyObject *args) {
    setupOpencv();
    // return Py_None;
    Py_RETURN_NONE;
}

static PyObject * readImage_py(PyObject *self, PyObject *args) {
    uint8_t *imageData = readImage();
    size_t imageSize = resize_height*resize_width*3;
    PyObject *list = PyList_New(imageSize);
    for(size_t i = 0; i < imageSize; i++) {
        PyList_SetItem(list, i, Py_BuildValue("B", imageData[i])); // B (int) [unsigned char]
    }
    return list;
}

static PyObject * readIndicators_py(PyObject *self, PyObject *args) {
    double *indicators = readIndicators();
    PyObject *list = PyList_New(16);
    for(size_t i = 0; i < 16; i++) {
        PyList_SetItem(list, i, Py_BuildValue("d", indicators[i])); // d (float) [double]
    }
    free(indicators);
    return list;
}

static PyObject * controller_py(PyObject *self, PyObject *args) {
    PyObject *pyList;
    double *indicators = (double *) calloc(INDICATORS_SIZE, sizeof(double));
    if (!PyArg_ParseTuple(args, "O", &pyList)) return NULL; // Warning. TODO: any prob?
    for(Py_ssize_t i = 0; i < INDICATORS_SIZE; i++) {
        double d = PyFloat_AsDouble(PyList_GetItem(pyList, i)); // Returns -1.0 on error. Use PyErr_Occurred() to disambiguate.
        if (PyErr_Occurred()) return NULL;
        indicators[i] = d;
    }
    controller(indicators);
    free(indicators);
    Py_RETURN_NONE;
}

///////////////////
// Python Module //
///////////////////

static struct PyMethodDef drive_mets[] = {
	{
        "test_shared_memory", testSharedMemory_py, METH_VARARGS,
        "Reads shared memory."
    },
    {
        "setup_shared_memory", setupSharedMemory_py, METH_VARARGS,
        "Setup shared memory."
    },
    {
        "close_shared_memory", closeSharedMemory_py, METH_VARARGS,
        "Close shared memory."
    },
    {
        "is_written", isWritten_py, METH_VARARGS,
        "Indicates whether torcs has updated the contents on shared memory."
    },
    {
        "write", write_py, METH_VARARGS,
        "Updates write flag on shared memory."
    },
    {
        "is_paused", isPaused_py, METH_VARARGS,
        "Indicates whether torcs should update the contents on shared memory."
    },
    {
        "pause", pause_py, METH_VARARGS,
        "Updates pause flag on shared memory."
    },
    {
        "is_controlling", isControlling_py, METH_VARARGS,
        "Control flag."
    },
    {
        "set_control", setControl_py, METH_VARARGS,
        "Control flag."
    },
    {
        "setup_opencv", setupOpencv_py, METH_VARARGS,
        "Setup opencv."
    },
    {
        "read_image", readImage_py, METH_VARARGS,
        "Read image from shared memory."
    },
    {
        "read_indicators", readIndicators_py, METH_VARARGS,
        "Read indicators from shared memory."
    },
    {
        "controller", controller_py, METH_VARARGS,
        "Outputs control commands given indicators"
    },
	{NULL, NULL, 0, NULL} /* Sentinel */
};

char drive_doc[] = "Drive module that interfaces with torcs.";

PyModuleDef drive_mod = {
	PyModuleDef_HEAD_INIT,
    "drive",   /* name of module */
    drive_doc, /* module documentation, may be NULL */
    -1,
    drive_mets
};

PyMODINIT_FUNC PyInit_drive(void) {
	return PyModule_Create(&drive_mod);
}
