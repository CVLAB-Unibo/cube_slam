cmake_minimum_required(VERSION 3.0.0)
SET(CMAKE_BUILD_TYPE Release)
project(detect_3d_cuboid)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.2 REQUIRED)

set(CUBOID_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/object_3d_util.cpp
    ${PROJECT_SOURCE_DIR}/matrix_utils.cpp
    ${PROJECT_SOURCE_DIR}/box_proposal_detail.cpp
)

add_library(${PROJECT_NAME} SHARED ${CUBOID_SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${PROJECT_SOURCE_DIR}
                           ${Eigen_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}  
)
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBRARIES}
)
