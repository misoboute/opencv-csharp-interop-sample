#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

std::vector<std::string> ProcessFrame(const cv::Mat& image);
