#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "json.h"

using namespace std;
using namespace cv;
//using namespace boost::property_tree;
void parseValue(const Json::Value &root, map<string, vector<vector<float> > > &result_map);
//void handelKey(const int &key);
void render_AI_14parts(Mat &img, const vector<vector<float> > &keypoints, int kind);

int main() {
    string gt_file = "";
    string pre_file = "";
    string image_absolute_path = "/mnt/sda1/yihongwei/dataset/AIChallenger/"
            "ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911";

    ifstream gt_ifs, pre_ifs;
    gt_ifs.open("/mnt/sda1/yihongwei/dataset/AIChallenger/"
                        "ai_challenger_keypoint_validation_20170911/"
                        "keypoint_validation_annotations_20170911.json");
    assert(gt_ifs.is_open());
    pre_ifs.open("/mnt/sda1/yihongwei/dataset/AIChallenger/"
                         "ai_challenger_keypoint_validation_20170911/"
                         "result.json");
    assert(pre_ifs.is_open());
    Json::Reader gt_reader, pre_reader;
    Json::Value gt_root, pre_root;
    if(!gt_reader.parse(gt_ifs, gt_root) || !pre_reader.parse(pre_ifs, pre_root)){
        return -1;
    }

    // parse json
    cout << "start parse json" << endl;
    map<string, vector<vector<float> > > gt_map, pre_map;
    parseValue(gt_root, gt_map);
    parseValue(pre_root, pre_map);

    // traversal pre_map
    map<string, vector<vector<float> > >::iterator map_iter = pre_map.begin();
    //for(map_iter = pre_map.begin(); map_iter != pre_map.end(); map_iter++){
    int kind = 1 ; //1 : predict annotation; 2: with gt annotation;
    while(map_iter != pre_map.end()){
        string image_id = map_iter->first;
        vector<vector<float> > annot_vec = map_iter->second;
        cout << "Pre image id: " << image_id << " has annotations " << annot_vec.size() <<endl;

        vector<vector<float> > gt_annot_vet;
        auto search = gt_map.find(image_id);
        if(search != gt_map.end()){
            gt_annot_vet = gt_map[image_id];
            cout << "GT image id: " << image_id << " has annotations " << gt_annot_vet.size() <<endl;
        }
        else{
            cout << "GT has not image" << image_id <<endl;
        }

        //show image
        string img_path = image_absolute_path + "/" + image_id + ".jpg";
        cout << "show image :" << img_path << endl;
        Mat origin_img = imread(img_path);
        cout << "size :" << origin_img.cols << ", " << origin_img.rows <<endl;
        switch(kind){
            case 1:
                //render pre
                render_AI_14parts(origin_img, annot_vec, 1);
                break;
            case 2:
                // render pre and gt
                render_AI_14parts(origin_img, annot_vec, 1);
                render_AI_14parts(origin_img, gt_annot_vet, 2); // for add gt keypoints
                break;
            default:
                break;
        }

        imshow("annotation", origin_img); // render_image
        //waitKey
        char key = waitKey(0);
        cout << "key " << key << endl;
        //HandleKey
        switch(key){
            case 110: // next image
                map_iter++;
                cout << "key n" << endl;
                break;
            case 98: //
                if(map_iter != pre_map.begin())
                    map_iter--;
                cout << "key b" <<endl;
                break;
            case 49 : // add gt
                kind = 1;
                cout << "kind 1" <<endl;
                break;
            case 50 : // remove gt
                kind = 2;
                cout << "kind 2 " << endl;
                break;
            default:
                break;
        }

    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}

void parseValue(const Json::Value &root, map<string, vector<vector<float> > > &result_map) {
    for (int i = 0; i < root.size(); i++) {
        string key = root[i]["image_id"].asString();
        Json::Value annot_list = root[i]["keypoint_annotations"];
        vector<vector<float>> annotations;
        for (int j = 0; j < annot_list.size(); j++) {
            string human = "human" + to_string(j + 1);
            vector<float> one_anno;
            for (int k = 0; k < annot_list[human].size(); k++) {
                one_anno.push_back(annot_list[human][k].asFloat());
            }
            annotations.push_back(one_anno);
        }
        result_map[key] = annotations;
    }
    cout << "parse Image cnt: " << result_map.size() << endl;
}


void render_AI_14parts(Mat &img, const vector<vector<float> > &keypoints, int kind){
    float radius = 3*img.rows / 300.0f;
    float alpha = 0.5;
    int color[27] = {
            255,   0, 0,
            255, 170, 0,
            170, 255, 0,
            0, 255, 0,
            0, 255, 170,
            0, 170, 255,
            0, 0,   255,
            170, 0,   255,
            255, 0,   170,
//            255, 185, 15,
//            205, 155, 155,
//            139, 71, 38,
//            205, 170, 125
    };
    int width = img.cols;
    int height = img.rows;
    for(int i = 0; i < keypoints.size(); i++){
        vector<float> key_anno = keypoints[i];
        for (int j = 0; j < 14; j++){
            int x = round(key_anno[3*j]);
            int y = round(key_anno[3*j + 1]);
            float score = round(key_anno[3*j + 2]);
            int label = round(key_anno[3*j + 2]);
            if (label == 0)
                continue;
            for (int x_radius = x - radius; x_radius < x + radius; x_radius++)
                for(int y_radius = y - radius; y_radius < y + radius; y_radius++)
                    if(x_radius < width && y_radius < height && x_radius > 0 && y_radius > 0){
                        switch(kind){
                            case 2:
                                if(label == 1){
                                    img.at<Vec3b>(y_radius,x_radius)[0] = 0;
                                    img.at<Vec3b>(y_radius,x_radius)[1] = 0;
                                    img.at<Vec3b>(y_radius,x_radius)[2] = 0;
                                }
                                else{
                                    img.at<Vec3b>(y_radius,x_radius)[0] = 255;
                                    img.at<Vec3b>(y_radius,x_radius)[1] = 255;
                                    img.at<Vec3b>(y_radius,x_radius)[2] = 255;
                                }

                                break;

                            case 1:
                            default:
                                float b, g, r;
                                b = img.at<Vec3b>(y_radius,x_radius)[0];
                                g = img.at<Vec3b>(y_radius,x_radius)[1];
                                r = img.at<Vec3b>(y_radius,x_radius)[2];

                                b = (1-alpha) * b + alpha * color[(j % 9)*3 + 2];
                                g = (1-alpha) * g + alpha * color[(j % 9)*3 + 1];
                                r = (1-alpha) * r + alpha * color[(j % 9)*3 ];

                                img.at<Vec3b>(y_radius,x_radius)[0] = b;
                                img.at<Vec3b>(y_radius,x_radius)[1] = g;
                                img.at<Vec3b>(y_radius,x_radius)[2] = r;
                                break;
                        }
                    }
        }

    }
}

/*
void handelKey(const int &key){
    if (key == " "){

    }
}
 */