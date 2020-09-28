cmake_minimum_required(VERSION 3.0.0)
SET(CMAKE_BUILD_TYPE Release)
project(orb_object_slam)

find_package(OpenCV 3.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL 1.2 REQUIRED)

set(DBOW2_PATH ../../thirdparty/DBoW2)
set(G2O_PATH ../../thirdparty/g2o)
set(CUBOID_PATH ../detect_3d_cuboid)
set(SETTINGS_PATH ../multi_settings)

ADD_SUBDIRECTORY(${DBOW2_PATH} ${DBOW2_PATH}/build)
ADD_SUBDIRECTORY(${G2O_PATH} ${G2O_PATH}/build)
ADD_SUBDIRECTORY(${CUBOID_PATH} ${CUBOID_PATH}/build)
ADD_SUBDIRECTORY(${SETTINGS_PATH} ${SETTINGS_PATH}/build)

set(ORB_OBJ_SLAM_FILES 
    ${PROJECT_SOURCE_DIR}/System.cc
    ${PROJECT_SOURCE_DIR}/Tracking.cc
    ${PROJECT_SOURCE_DIR}/Tracking_util.cc
    ${PROJECT_SOURCE_DIR}/LocalMapping.cc
    ${PROJECT_SOURCE_DIR}/LoopClosing.cc
    ${PROJECT_SOURCE_DIR}/ORBextractor.cc
    ${PROJECT_SOURCE_DIR}/ORBmatcher.cc
    ${PROJECT_SOURCE_DIR}/FrameDrawer.cc
    ${PROJECT_SOURCE_DIR}/Converter.cc
    ${PROJECT_SOURCE_DIR}/MapPoint.cc
    ${PROJECT_SOURCE_DIR}/MapObject.cc
    ${PROJECT_SOURCE_DIR}/KeyFrame.cc
    ${PROJECT_SOURCE_DIR}/Map.cc
    ${PROJECT_SOURCE_DIR}/MapDrawer.cc
    ${PROJECT_SOURCE_DIR}/Optimizer.cc
    ${PROJECT_SOURCE_DIR}/PnPsolver.cc
    ${PROJECT_SOURCE_DIR}/Frame.cc
    ${PROJECT_SOURCE_DIR}/KeyFrameDatabase.cc
    ${PROJECT_SOURCE_DIR}/Sim3Solver.cc
    ${PROJECT_SOURCE_DIR}/Initializer.cc
    ${PROJECT_SOURCE_DIR}/Viewer.cc
    ${PROJECT_SOURCE_DIR}/Parameters.cpp
    ${PROJECT_SOURCE_DIR}/g2o_Object.cpp
)

add_library(${PROJECT_NAME} SHARED ${ORB_OBJ_SLAM_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${PROJECT_SOURCE_DIR}
                           ${PROJECT_SOURCE_DIR}/..
                           ${OpenCV_INCLUDE_DIRS}
                           ${EIGEN_INCLUDE_DIR}
                           ${Pangolin_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS}
                           ${DBOW2_PATH}
                           ${G2O_PATH}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBS}
                      ${Pangolin_LIBRARIES}
                      ${PCL_LIBRARIES}
                      DBoW2
                      g2o
                      detect_3d_cuboid
                      multi_settings
)
