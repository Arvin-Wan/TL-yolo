#include <torch/torch.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <torch/script.h>
#include <memory>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>
#include <map>
#include <iomanip>
#include <math.h>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <ctime>
#include <io.h>
#include <dirent.h>
#include <tuple>
#include "../include/cxxopts.hpp"
#include <fstream>


using namespace cv;
using namespace torch;
using namespace std;

torch::Tensor xywh2xyxy(torch::Tensor x){

    auto y = torch::zeros_like(x);
    auto z = y.accessor<float, 2>();
    y.select(1,0) = x.select(1, 0) - x.select(1, 2).div(2);
    y.select(1,1)= x.select(1, 1) - x.select(1, 3).div(2);
    y.select(1,2)= x.select(1, 0) + x.select(1, 2).div(2);
    y.select(1,3) = x.select(1, 1) + x.select(1, 3).div(2);

    return y;

}


torch::Tensor non_max_suppression(torch::Tensor prediction, torch::DeviceType device_type, float conf_thres, float iou_thres){
    int batch_size = prediction.size(0);
    int nc = prediction.size(2) -5;
    int num_anchors = prediction.size(1);
    torch::Tensor result;


 
    //shape [1, 25200]
    //tensor.select(dim_index,index)
    //tensor c = torch.ge(a,b)  => boolen(a > b)
    auto conf_tensor = prediction.select(2,4) > conf_thres;
    conf_tensor = conf_tensor.unsqueeze(2);
    //auto conf_tensor = prediction.select(2,4).ge(conf_thres).unsqueeze(2);
 
    std::vector<torch::Tensor> output;
    output.push_back(torch::ones({batch_size,6}));
    for (int batch_i=0; batch_i < batch_size; batch_i++){
        auto x = torch::masked_select(prediction[batch_i], conf_tensor[batch_i]).view({-1, nc + 5});

        if(0 == x.size(0)){
            continue;
        }

        x.slice(1,5, nc+5) *= x.select(1,4).unsqueeze(1);

        torch::Tensor box = xywh2xyxy(x.slice(1,0,4));

        // [best class only] get the max classes score at each result (e.g. elements 5-84)
        std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(x.slice(1, 5, 5 + nc), 1);

        // class score
        auto max_conf_score = std::get<0>(max_classes);
        // index
        auto max_conf_index = std::get<1>(max_classes);

        max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
        max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);


        // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
        x = torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

        // for batched NMS
        constexpr int max_wh = 4096;
        auto c=x.slice(1, 5, 6) * max_wh;
        auto boxes = x.slice(1,0,4) +c;
        auto scores = x.slice(1,4,5);


        std::vector<cv::Rect> offset_box_vec;
        std::vector<float> score_vec;

        for (int i =0; i < boxes.size(0); i++){
            float left_x = boxes[i][0].item().toFloat();
            float left_y = boxes[i][1].item().toFloat();
            float right_x = boxes[i][2].item().toFloat();
            float right_y = boxes[i][3].item().toFloat();
            
            offset_box_vec.push_back(
                cv::Rect(left_x, left_y,
                        right_x, right_y)
            );
            score_vec.push_back(scores[i][0].item().toFloat());
        }
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

        result = x[nms_indices[0]];

    }

    return result;

}



torch::Tensor get_script_input(cv::Mat& img, torch::DeviceType device_type){
    /*  Transfer BGR to RGB, Transpose HWC to BCHW , 
    and rescale the range of image from 0-255 to 0-1.
    Args:
        img :  cv::Mat format, the output of func img_resize.
        device_type: cpu ,represents that the folowing operations are performed on the cpu
    
    returns:
        torch::Tensor img_tensor;
    */
    
    const int height = img.rows;
    const int width = img.cols;
    const int channels = 3;
    cv::cvtColor(img,img, cv::COLOR_BGR2RGB);

    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    auto img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}).to(device_type);

    img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();

    return img_tensor;
}

torch::Tensor yolo_detect(torch::jit::script::Module model, Mat img, float conf_thres, float iou_thres, torch::DeviceType device_type){
    /* detect function
    Args:
        model: model after loaded the model.pt/model.torchscript weights file
        img: opencv imread
        conf_thres: confidence threshold value between[0,1]
        iou_thres: iou threshold value between [0,1]
        device_type: cpu
    
    Returns:
        result: tensor, 
            like this : tensor([x1,y1,x2,y2,score,class_id]), 
            (x1,y1) is the point of top-left bbox,(x2,y2) is the point of botton-right bbox,
            scode is confidence, class_id is the index of classes.

     */
    
    cv::Size size = cv::Size(640,640);
    cv::resize(img, img, size);

    torch::Tensor input = get_script_input(img, device_type);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    auto start = std::chrono::high_resolution_clock::now();
    //std::vector<c10::IValue> input_vector = input_value; 
    auto outputs = model(inputs).toTuple()->elements()[0].toTensor();
    

    torch::Tensor result = non_max_suppression(outputs, device_type, conf_thres, iou_thres);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "reference cost :" << duration.count() << "ms" << endl;
    return result;
}

int main(int argc, const char* argv[]){
    
    torch::DeviceType device_type;
    device_type = torch::kCPU;
    float conf_thres = 0.25;
    float iou_thres = 0.45;
    map<string, torch::Tensor> result_map;

    //weights path
    string model_path = "../../model/best.torchscript";
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.eval();
    string image_path = argv[1];
    Mat img = imread(image_path, cv::IMREAD_COLOR);
    
    if (img.empty()) {
        cout << "Please check the image or the path" << endl;
        return -1;
    }
    auto result = yolo_detect(model, img, conf_thres, iou_thres, device_type);
    result_map["bbox"]= result.slice(0,0,4);
    result_map["score"]= result.slice(0,4,5);
    result_map["class_id"]= result.slice(0,5,6);

    cout << "result map is " << result_map << endl;
    return 0;
}