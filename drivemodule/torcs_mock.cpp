#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <sys/shm.h>

#define image_width 640
#define image_height 480

struct shared_use_st  
{  
    int written;
    uint8_t data[image_width*image_height*3];
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

int* pwritten = NULL;
uint8_t* pdata = NULL;
int* pcontrol = NULL;
int* ppause = NULL;

double* psteerCmd_ghost = NULL;
double* paccelCmd_ghost = NULL;
double* pbrakeCmd_ghost = NULL;

double* pspeed_ghost = NULL;
double* ptoMiddle_ghost = NULL;
double* pangle_ghost = NULL;

double* pfast_ghost = NULL;

double* pdist_L_ghost = NULL;
double* pdist_R_ghost = NULL;

double* ptoMarking_L_ghost = NULL;
double* ptoMarking_M_ghost = NULL;
double* ptoMarking_R_ghost = NULL;

double* pdist_LL_ghost = NULL;
double* pdist_MM_ghost = NULL;
double* pdist_RR_ghost = NULL;

double* ptoMarking_LL_ghost = NULL;
double* ptoMarking_ML_ghost = NULL;
double* ptoMarking_MR_ghost = NULL;
double* ptoMarking_RR_ghost = NULL;

void *shm = NULL;

int
main(int argc, char *argv[])
{
    struct shared_use_st *shared = NULL;
    int shmid;
    char buffer[50];
    
    // establish memory sharing 
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
    printf("\n********** Memory sharing started, attached at %p **********\n \n", shm);  
    
    // set up shared memory 
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

    if(shmctl(shmid, IPC_RMID, 0) == -1) { // TODO: Only called on one side
        fprintf(stderr, "shmctl(IPC_RMID) failed\n");  
        exit(EXIT_FAILURE);  
    }
    printf("\n********** Memory sharing stopped. Good Bye! **********\n");    
    exit(EXIT_SUCCESS); 
}