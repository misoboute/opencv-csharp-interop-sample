// DetectCamLib.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"

#include "DetectCam.h"

#include <torch/script.h>

using namespace cv;
using namespace std;

std::vector<std::string> ProcessFrame(const cv::Mat& image)
{
    // actual function implementation
    int k = 0;
    cv::Mat croppedimage;
    cv::Mat finalcropped;
    string filename;
    Mat result_image;
    vector<string> listName;
    Module module = torch::jit::load("D:/Project/libfacedetection/example/converted.pt");


    int* pResults = NULL;


    unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
    if (!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return listName;
    }


    TickMeter cvtm;
    cvtm.start();


    pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);


    int face_num = (pResults ? *pResults : 0);

    if (*pResults != 0)
    {
        result_image = image.clone();

        for (int i = 0; i < face_num; i++)
        {
            try
            {
                short* p = ((short*)(pResults + 1)) + 142 * i;
                int confidence = p[0];
                int x = p[1];
                int y = p[2];
                int w = p[3];
                int h = p[4];

                char sScore[256];


                if (confidence >= 95)
                {

                    //////////////////////////////////////////////////////////////////////////////
                    ////////////// Rotate and Crop
                    //////////////////////////////////////////////////////////////////////////////

                    short angle = Face_rotate(p);

                    cv::Rect rc = AlignCordinates(x, y, w, h, result_image.cols, result_image.rows);

                    cv::Rect myroi(x, y, w, h);
                    cv::Rect newroi((x - rc.x) / 2, (y - rc.y) / 2, w, h);

                    croppedimage = result_image(rc);
                    //imshow("1", croppedimage);

                    croppedimage = croppedimage.clone();
                    croppedimage = rotate(croppedimage, (angle));


                    //imshow("Rotate", croppedimage);


                    croppedimage = croppedimage(newroi).clone();

                    finalcropped = Mat(112, 112, croppedimage.type());
                    //imshow("dst", croppedimage);


                    cv::resize(croppedimage, finalcropped, finalcropped.size());
                    //imshow("resize", finalcropped);
                    Mat flipimage;
                    flip(finalcropped, flipimage, 1);



                    torch::Tensor img_tensor = torch::from_blob(finalcropped.data, { finalcropped.rows,finalcropped.cols ,3 }, torch::kByte);
                    torch::Tensor img_tensor_flip = torch::from_blob(flipimage.data, { flipimage.rows, flipimage.cols, 3 }, torch::kByte);

                    //torch::Tensor img_tensor_final = img_tensor + img_tensor_flip;

                    img_tensor = img_tensor.to(at::kFloat).div(255).unsqueeze(0);
                    img_tensor = img_tensor.sub_(0.5);
                    img_tensor = img_tensor.permute({ 0,3,1,2 });

                    img_tensor_flip = img_tensor_flip.to(at::kFloat).div(255).unsqueeze(0);
                    img_tensor_flip = img_tensor_flip.sub_(0.5);
                    img_tensor_flip = img_tensor_flip.permute({ 0,3,1,2 });



                    at::Tensor output_org = module.forward({ img_tensor }).toTensor();
                    at::Tensor output_flip = module.forward({ img_tensor_flip }).toTensor();

                    std::vector<double> out;


                    for (int i = 0; i < 512; i++)
                    {
                        out.push_back(output_org[0][i].item().to<double>() + output_flip[0][i].item().to<double>());
                    }

                    out = l2_norm(out);




                    std::ifstream file("D:/Project/libfacedetection/example/facebank.json");
                    json object = json::parse(file);




                    double min_dis = 1000;
                    std::string min_name;

                    for (auto& x : object.items()) {
                        auto dataSize = std::size(x.value());

                        std::vector<double> vec1 = x.value();



                        double res = cosine_similarity_vectors(vec1, out);
                        res = (res * -1) + 1;
                        //double res = distance(vec1, out);


                        if (res <= min_dis) {
                            min_dis = res;
                            min_name = x.key();
                        }
                    }




                    std::cout << "One Frame   " << min_name << " " << min_dis << std::endl;


                    if (min_dis < 0.8) {

                        listName.push_back(min_name);
                    }
                    else
                    {
                        listName.push_back("Unknown");
                    }
                }

                else
                {
                    listName.push_back("conf_low");

                }


            }
            catch (const std::exception& ex)
            {
                cout << "NASHOD" << endl;

                //std::cout << ex.what();
            }



        }
    }


    else
    {
        listName.push_back("No_Body");
    }
    cvtm.stop();



    //printf("time = %gms\n", cvtm.getTimeMilli());
    //printf("%d faces detected.\n", (pResults ? *pResults : 0));
    free(pBuffer);

    return listName;
}
