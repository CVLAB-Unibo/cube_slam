#pragma once

#include <iostream>

#define ROS_ERROR_STREAM(X) (std::cout << X << "\n")
#define ROS_WARN_STREAM(X) ROS_ERROR_STREAM(X)