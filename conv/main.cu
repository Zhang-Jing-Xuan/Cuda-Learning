#include <stdio.h>

static void HandleError(cudaError_t err,const char * file,int line){
    if(err!=cudaSuccess){
        printf("%s in %s at line %d\n",cudaGetErrorString(err),file,line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))
int main(){
    int width=10;
    int height=10;
    float *img=new float[width*height];
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            img[j+i*width]=(i+j)%256;
        }
    }

    int kernelSize=3;
    float *kernel=new float[kernelSize*kernelSize];
    for(int i=0;i<kernelSize*kernelSize;i++){
        kernel[i]=i%kernelSize-1;
    }

    float *imgGpu,*kernelGpu,*resultGpu;

    HANDLE_ERROR(cudaMalloc((void**)&imgGpu,width*height*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&kernelGpu,kernelSize*kernelSize*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&resultGpu,width*height*sizeof(float)));

    HANDLE_ERROR(cudaMemcpy());
    //Visualization
    printf("img:\n");
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            printf("%2.0f ",img[j+i*width]);
        }
        puts("");
    }
    printf("kernel:\n");
    for(int i=0;i<kernelSize;i++){
        for(int j=0;j<kernelSize;j++){
            printf("%2.0lf ",kernel[i*kernelSize+j]);
        }
        puts("");
    }
    return 0;
}