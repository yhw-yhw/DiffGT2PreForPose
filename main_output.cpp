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

int main(int argc, char ** argv) {
    string gt_file = "";
    string pre_file = "";
    string image_absolute_path = "";
    string save_dir_path = "";
    int kind_flag = 0;
    bool is_show = false;
    // bin absolute_dir_path save_path gt_json/pre_json kind(1 for gt, 2 for pre)
    // bin absolute_dir_path save_path pre_json gt_json kind(3)
    if(argc < 5 || argc > 6){
        cout << "please input bin absolute_dir_path "
                "save_path gt_json/pre_json kind(1 for gt, 2 for pre) \n"
             "Or bin absolute_dir_path save_path pre_json gt_json kind(3)" <<endl;
        return 0;
    }
    else if (argc == 5){
        image_absolute_path = string(argv[1]);
        save_dir_path = string(argv[2]);
        kind_flag = atoi(argv[4]);
        if(kind_flag == 1){
            gt_file = string(argv[3]);
        }
        else if(kind_flag == 2) {
            pre_file = string(argv[3]);
        }
    }
    else if(argc == 6){
        image_absolute_path = string(argv[1]);
        save_dir_path = string(argv[2]);
        kind_flag = atoi(argv[5]);
        gt_file = string(argv[4]);
        pre_file = string(argv[3]);
    }



    ifstream gt_ifs, pre_ifs;
    if(kind_flag == 1){
        gt_ifs.open(gt_file);
        assert(gt_ifs.is_open());
    }
    else if(kind_flag == 2){
        pre_ifs.open(pre_file);
        assert(pre_ifs.is_open());
    }
    else if (kind_flag == 3){
        pre_ifs.open(pre_file);
        assert(pre_ifs.is_open());
        gt_ifs.open(gt_file);
        assert(gt_ifs.is_open());
    }
    else{
        cout << "wrong input kind" << endl;
        return 0;
    }

    Json::Reader gt_reader, pre_reader;
    Json::Value gt_root, pre_root;
    if(kind_flag == 1 && !gt_reader.parse(gt_ifs, gt_root)){
        return -1;
    }
    else if(kind_flag == 2 && !pre_reader.parse(pre_ifs, pre_root)){
        return -1;
    }
    else if(kind_flag == 3 && (!gt_reader.parse(gt_ifs, gt_root) || !pre_reader.parse(pre_ifs, pre_root))){
        return -1;
    }

    // parse json
    cout << "start parse json" << endl;
    map<string, vector<vector<float> > > gt_map, pre_map;
    map<string, vector<vector<float> > >::iterator map_iter;
    int count = 0;
    switch(kind_flag){
        case 1:
            parseValue(gt_root, gt_map);
            // traversal pre_map
            map_iter = gt_map.begin();
            //for(map_iter = pre_map.begin(); map_iter != pre_map.end(); map_iter++){
          //  int kind = 1 ; //1 : predict annotation; 2: with gt annotation;
            while(map_iter != gt_map.end()) {
                string image_id = map_iter->first;
                vector<vector<float> > annot_vec = map_iter->second;
                cout << "GT image id: " << image_id << " has annotations " << annot_vec.size() << endl;

                string img_path = image_absolute_path + "/" + image_id + ".jpg";
                cout << "show image :" << img_path << endl;
                Mat origin_img = imread(img_path);
                cout << "size :" << origin_img.cols << ", " << origin_img.rows <<endl;
                //start render gt
                render_AI_14parts(origin_img, annot_vec, 2);
                string save_file = save_dir_path + "/" + image_id + ".jpg";
                if(is_show == true){
                    imshow("GT_annotation", origin_img);
                    waitKey(30);
                }

                imwrite(save_file, origin_img);
                count++;
                cout << "write frame " << count << " " << save_file << endl;
                map_iter++;
            }
            break;

        case 2:
            parseValue(pre_root, pre_map);
            // traversal pre_map
            map_iter = pre_map.begin();
            //for(map_iter = pre_map.begin(); map_iter != pre_map.end(); map_iter++){
            //  int kind = 1 ; //1 : predict annotation; 2: with gt annotation;
            while(map_iter != pre_map.end()) {
                string image_id = map_iter->first;
                vector<vector<float> > anno_vec = map_iter->second;
                cout << "GT image id: " << image_id << " has annotations " << anno_vec.size() << endl;

                string img_path = image_absolute_path + "/" + image_id + ".jpg";
                cout << "show image :" << img_path << endl;
                Mat origin_img = imread(img_path);
                cout << "size :" << origin_img.cols << ", " << origin_img.rows <<endl;
                //start render gt
                render_AI_14parts(origin_img, anno_vec, 1);
                string save_file = save_dir_path + "/" + image_id + ".jpg";
                if(is_show == true) {
                    imshow("Pre_annotation", origin_img);
                    waitKey(30);
                }
                imwrite(save_file, origin_img);
                count++;
                cout << "write frame " << count << " " << save_file << endl;
                map_iter++;
            }
            break;

        case 3:
            parseValue(gt_root, gt_map);
            parseValue(pre_root, pre_map);

            map_iter = pre_map.begin();
            //for(map_iter = pre_map.begin(); map_iter != pre_map.end(); map_iter++){
            while(map_iter != pre_map.end()) {
                string image_id = map_iter->first;
                vector<vector<float> > annot_vec = map_iter->second;
                cout << "Pre image id: " << image_id << " has annotations " << annot_vec.size() << endl;

                vector<vector<float> > gt_annot_vet;
                auto search = gt_map.find(image_id);
                if (search != gt_map.end()) {
                    gt_annot_vet = gt_map[image_id];
                    cout << "GT image id: " << image_id << " has annotations " << gt_annot_vet.size() << endl;
                } else {
                    cout << "GT has not image " << image_id << endl;
                    continue;
                }


                //show image
                string img_path = image_absolute_path + "/" + image_id + ".jpg";
                cout << "show image :" << img_path << endl;
                Mat origin_img = imread(img_path);
                cout << "size :" << origin_img.cols << ", " << origin_img.rows << endl;
                render_AI_14parts(origin_img, annot_vec, 1);
                render_AI_14parts(origin_img, gt_annot_vet, 2); // for add gt keypoints
                if(is_show == true) {
                    imshow("DiffGT2Pre annotation", origin_img); // render_image
                    waitKey(30);
                }
                string save_file = save_dir_path + "/" + image_id + ".jpg";
                imwrite(save_file, origin_img);
                count++;
                cout << "write frame " << count << " " << save_file << endl;
                map_iter++;
            }
            break;

        default:
            cout << "wrong input kind" << endl;
            break;
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
    float alpha = 0.3;
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