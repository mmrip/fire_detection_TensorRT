#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo/yolo.hpp"
#include "app_yolo/multi_gpu.hpp"
#include "opencv2/opencv.hpp"
using namespace std;

static const char* labels[] = {"fire"};

bool requires(const char* name);

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, Yolo::Type type, const string& model_name){

    auto engine = Yolo::create_infer(
        engine_file,                // engine file
        type,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
        deviceid,                   // gpu id
        0.25f,                      // confidence threshold
        0.45f,                      // nms threshold
        Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        256,                       // max objects
        false                       // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto type_name = Yolo::type_name(type);
    auto mode_name = TRT::mode_string(mode);

    shared_future<Yolo::BoxArray> box_array;
    cv::VideoCapture capture(0);
    cv::Mat frame;

    while (true){
        capture >> frame;
        auto begin_timer = iLogger::timestamp_now_float();
        box_array = engine->commit(frame);
        box_array.get();
        float inference_average_time = (iLogger::timestamp_now_float() - begin_timer);
        
        auto& image = frame;
        auto boxes  = box_array.get();
        
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto name    = labels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        INFO("Detect %d object, average: %.2f ms / image, FPS: %.2f", boxes.size(), inference_average_time, 1000 / inference_average_time);
        cv::imwrite("result.png", image);
        // cv::imshow("result", image);
    }
    engine.reset();
}

static void run(Yolo::Type type, TRT::Mode mode, const string& model){
    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== run %s %s %s ==================================", Yolo::type_name(type), mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 1;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }

    inference_and_performance(deviceid, model_file, mode, type, name);
}

int app_fire_detection(){
    run(Yolo::Type::V5, TRT::Mode::FP16, "fire");
}