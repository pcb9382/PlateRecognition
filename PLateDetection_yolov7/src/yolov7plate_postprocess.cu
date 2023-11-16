#include "yolov7plate_postprocess.h"
#define MAX_LANDMARK 20
static __device__ void yolov7plate_affine_project(float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void yolov7plate_decode_kernel(float* predict,int NUM_BOX_ELEMENT, int num_bboxes, int num_classes,int ckpt, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects)
{  

    //printf("NUM_BOX_ELEMENT:%d,num_bboxes:%d,num_classes:%d,ckpt%d,confidence_threshold%f,max_objects%d",NUM_BOX_ELEMENT,num_bboxes,num_classes,ckpt,confidence_threshold,max_objects);
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) //25200
    {
        return;
    }
    float*pitem= predict+(5 + num_classes+ckpt*3) * position;//每个线程处理一个车牌的19个参数
    float objectness = pitem[4];                             //bbox置信度
    if(objectness < confidence_threshold)                    //小于该置信度的舍弃
    {
        return;
    }
    
    //多类别判断置信度和类别,找到置信度最大的类别
    float*class_confidence = pitem + 5;                     //第一个类别置信度
    float confidence= *class_confidence++;                  //先++，后*，第二个类别置信度
    int label= 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if(*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    confidence *= objectness;                               //最终的置信度是类别置信度*object的置信度                            
    if(confidence < confidence_threshold)
    {
        return;
    }
    //判断是否超过了最大人脸框的阈值,bbox个数存放在parray[0]
    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
    {
        return;
    }
    //printf("index %d max_objects %d\n", index,max_objects);
    float cx         = pitem[0];
    float cy         = pitem[1];
    float width      = pitem[2];
    float height     = pitem[3];
    
    //五个关键点
    float *landmarks = pitem+5+num_classes;
    float landmark_array[MAX_LANDMARK*2];
    for (int i = 0; i<ckpt; i++)
    {
        landmark_array[2*i]=landmarks[3*i];
        landmark_array[2*i+1]=landmarks[3*i+1];
    }

    //从cx,cy,w,h计算出(x1,y1,x2,y2)
    float left   = cx - width * 0.5f;        
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    //bbox还原到原图的位置
    yolov7plate_affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    yolov7plate_affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
    //landmark还原到原图的位置
    for(int i = 0; i<ckpt; i++)
    {
        yolov7plate_affine_project(invert_affine_matrix, landmark_array[2*i],landmark_array[2*i+1],&landmark_array[2*i],&landmark_array[2*i+1]); 
    }
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;            //1
    *pout_item++ = top;             //2
    *pout_item++ = right;           //3
    *pout_item++= bottom;           //4
    *pout_item++ = confidence;      //5
    *pout_item++ = label;           //6
    *pout_item++ = 1;               //7 1 = keep, 0 = ignore
    for(int i = 0; i<ckpt; i++)
    {
        *pout_item++=landmark_array[2*i];
        *pout_item++=landmark_array[2*i+1];
    }
}

static __device__ float yolov7plate_box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom)
{

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
    {
        return 0.0f;
    }
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void yolov7plate_nms_kernel(float* bboxes, int max_objects, float threshold,int NUM_BOX_ELEMENT)
{

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    //得到bbox数量
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
    {
        return;
    }
    //
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i)
    {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        //是同一个或者不是同一类，则跳过
        if(i == position || pcurrent[5] != pitem[5]) 
        {
            continue;
        }
        //置信度大于本次并且iou大于阈值，则pcurrent[6] = 0
        if(pitem[4] >= pcurrent[4])
        {
            if(pitem[4] == pcurrent[4] && i < position)
            {
                continue;
            }
            //iou
            float iou = yolov7plate_box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],pitem[0],pitem[1],pitem[2],pitem[3]);
            if(iou > threshold)
            {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 

void yolov7plate_decode_kernel_invoker(float* predict, int  NUM_BOX_ELEMENT,int num_bboxes,int num_classes,int ckpt, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream)
{
    int block = 256;
    int  grid =  ceil((num_bboxes+block-1) / (float)block);
    yolov7plate_decode_kernel<<<grid, block, 0, stream>>>(predict,NUM_BOX_ELEMENT, num_bboxes, num_classes,ckpt, confidence_threshold, invert_affine_matrix, parray, max_objects);
}

void yolov7plate_nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream,int NUM_BOX_ELEMENT)
{
    int block = max_objects<256? max_objects:256;
    int grid = ceil((max_objects+block-1) / (float)block);
    yolov7plate_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold,NUM_BOX_ELEMENT);
}
