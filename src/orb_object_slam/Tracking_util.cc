
#include "Tracking.h"

#include "Converter.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapObject.h"
#include "MapPoint.h"

// by me
#include "Parameters.h"
#include "detect_3d_cuboid/detect_3d_cuboid.h"
#include "detect_3d_cuboid/object_3d_util.h"
#include "ros_moc.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace Eigen;

namespace ORB_SLAM2
{

void Tracking::ReadAllObjecttxt()
{
    int total_img_ind = 10000;              // some large number
    all_offline_object_cubes.reserve(2000); // vector will double capacity automatically after full
    bool set_all_obj_probto_one = false;

    if (use_truth_trackid)
        ROS_WARN_STREAM("Read ground truth object tracking id");

    for (int img_couter = 0; img_couter < total_img_ind; img_couter++)
    {
        char frame_index_c[256];
        sprintf(frame_index_c, "%04d", img_couter); // format into 4 digit
        std::string pred_frame_obj_txts;

        pred_frame_obj_txts =
            data_folder + "/pred_3d_obj_matched_txt/" + frame_index_c + "_3d_cuboids.txt";
        if (use_truth_trackid)
            pred_frame_obj_txts = data_folder + "/pred_3d_obj_matched_tracked_txt/" +
                                  frame_index_c + "_3d_cuboids.txt";

        // 3d cuboid txts:  each row:  [cuboid_center(3), yaw, cuboid_scale(3), [x1 y1 w h]], prob
        int data_width = 12;
        if (use_truth_trackid)
            data_width = 13;
        Eigen::MatrixXd pred_frame_objects(5, data_width);
        if (read_all_number_txt(pred_frame_obj_txts, pred_frame_objects))
        {
            if (set_all_obj_probto_one)
                for (int ii = 0; ii < pred_frame_objects.rows(); ii++)
                    pred_frame_objects(ii, data_width - 1) = 1;

            all_offline_object_cubes.push_back(pred_frame_objects);

            if (!use_truth_trackid)
                if (pred_frame_objects.rows() > 0)
                    for (int ii = 0; ii < pred_frame_objects.rows(); ii++)
                        if (pred_frame_objects(ii, data_width - 1) < -0.1)
                            ROS_ERROR_STREAM("Read offline Bad object prob  "
                                             << pred_frame_objects(ii, data_width - 1)
                                             << "   frame  " << img_couter << "  row  " << ii);
        }
        else
        {
            ROS_WARN_STREAM("Totally read object txt num  " << img_couter);
            break;
        }
    }
}

void Tracking::SaveOptimizedCuboidsToTxt()
{
    ROS_WARN_STREAM("Save optimized cuboids into txts!!!");

    g2o::SE3Quat InitToGround_se3 = Converter::toSE3Quat(InitToGround);

    // directly record global object pose, into one txt
    vector<MapObject *> all_Map_objs = mpMap->GetAllMapObjects();
    std::string save_object_pose_txt = data_folder + "/orb_opti_pred_objs.txt";
    if (boost::filesystem::exists(save_object_pose_txt))
        boost::filesystem::remove(save_object_pose_txt);

    if (!whether_dynamic_object) // for static object, record final pose
    {
        int obj_counter = 0;
        save_final_optimized_cuboids.open(save_object_pose_txt.c_str());
        for (size_t i = 0; i < all_Map_objs.size(); i++)
        {
            MapObject *pMO = all_Map_objs[i];
            if (!pMO->isBad())
            {
                pMO->record_txtrow_id = obj_counter++;
                // transform to ground frame which is more visible.
                g2o::cuboid cube_pose_to_init = pMO->GetWorldPos();
                g2o::cuboid cube_pose_to_ground =
                    cube_pose_to_init.transform_from(InitToGround_se3); // absolute ground frame.
                if (build_worldframe_on_ground)
                    cube_pose_to_ground = cube_pose_to_init;
                save_final_optimized_cuboids << pMO->mnId << "  " << pMO->isGood << "  "
                                             << cube_pose_to_ground.toVector().transpose() << " "
                                             << "\n";
            }
        }
        save_final_optimized_cuboids.close();
        save_online_detected_cuboids.close();
    }

    if (whether_dynamic_object)
    {
        std::string save_object_velocity_txt = data_folder + "/orb_opti_pred_objs_velocity.txt";
        std::ofstream Logfile;
        Logfile.open(save_object_velocity_txt.c_str());
        ROS_ERROR_STREAM("save total object size   " << all_Map_objs.size());
        for (size_t i = 0; i < all_Map_objs.size(); i++)
        {
            MapObject *pMO = all_Map_objs[i];
            {
                for (map<KeyFrame *, Eigen::Vector2d, cmpKeyframe>::iterator mit =
                         pMO->velocityhistory.begin();
                     mit != pMO->velocityhistory.end(); mit++)
                {
                    Logfile << pMO->truth_tracklet_id << "  " << mit->first->mnFrameId << "    "
                            << mit->second.transpose() << "\n";
                }
            }
        }
        Logfile.close();
    }

    // record object pose in each frame, into different txts
    if ((scene_unique_id == kitti) && whether_dynamic_object)
    {
        // for KITTI  for each keyframe, save its observed optimized cuboids into txt.
        const vector<KeyFrame *> all_keyframes = mpMap->GetAllKeyFrames(); // not sequential

        std::string kitti_saved_obj_dir = data_folder + "/orb_frame_3d/";
        if (whether_dynamic_object)
            kitti_saved_obj_dir = data_folder + "/orb_obj_3d/";

        if (boost::filesystem::exists(kitti_saved_obj_dir))
            boost::filesystem::remove_all(kitti_saved_obj_dir);

        boost::filesystem::create_directories(kitti_saved_obj_dir);

        for (size_t i = 0; i < all_keyframes.size(); i++)
        {
            KeyFrame *kf = all_keyframes[i];

            char sequence_frame_index_c[256];
            sprintf(sequence_frame_index_c, "%04d", (int)kf->mnFrameId);
            std::string save_object_ba_pose_txt = kitti_saved_obj_dir + sequence_frame_index_c +
                                                  "_orb_3d_ba.txt"; // object pose after BA
            ofstream Logfile;
            Logfile.open(save_object_ba_pose_txt.c_str());
            std::string save_object_asso_pose_txt =
                kitti_saved_obj_dir + sequence_frame_index_c +
                "_orb_3d_asso.txt"; // object pose before BA, just using association
            ofstream Logfile2;
            Logfile2.open(save_object_asso_pose_txt.c_str());
            g2o::SE3Quat frame_pose_to_init =
                Converter::toSE3Quat(kf->GetPoseInverse()); // camera to init world

            for (size_t j = 0; j < kf->cuboids_landmark.size(); j++)
            {
                MapObject *pMO = kf->cuboids_landmark[j];

                if (!pMO)
                {
                    continue;
                }

                g2o::cuboid cube_global_pose;
                if (whether_dynamic_object) // get object pose at this frame.
                {
                    if (pMO->allDynamicPoses.count(kf))
                        cube_global_pose = pMO->allDynamicPoses[kf].first;
                    else
                        continue;
                }
                else
                    cube_global_pose = pMO->GetWorldPos();

                g2o::cuboid cube_to_camera = cube_global_pose.transform_to(
                    frame_pose_to_init); // measurement in local camera frame
                g2o::cuboid cube_to_local_ground =
                    cube_to_camera.transform_from(InitToGround_se3); // some approximation

                int object_ID = pMO->GetIndexInKeyFrame(
                    kf); // obj index in all raw frame cuboids. -1 if bad deleted object
                Logfile << cube_to_local_ground.toMinimalVector().transpose() << "    " << object_ID
                        << "   " << pMO->truth_tracklet_id << "\n";

                if (!whether_dynamic_object) // the cube before optimized!!!
                {
                    cube_to_camera = pMO->pose_noopti.transform_to(frame_pose_to_init);
                    cube_to_local_ground = cube_to_camera.transform_from(InitToGround_se3);
                    Logfile2 << cube_to_local_ground.toMinimalVector().transpose() << "    "
                             << object_ID << "\n";
                }
            }
            Logfile.close();
            Logfile2.close();
        }
    }

    return;
}

void Tracking::DetectCuboid(KeyFrame *pKF)
{
    cv::Mat pop_pose_to_ground; // pop frame pose to ground frame.  for offline txt, usually local
                                // ground.  for online detect, usually init ground.
    std::vector<ObjectSet> all_obj_cubes; // in ground frame, no matter read or online detect
    std::vector<Vector4d> all_obj2d_bbox;
    std::vector<double> all_box_confidence;
    std::vector<int> truth_tracklet_ids;

    if (whether_read_offline_cuboidtxt)
    {
        // saved object txt usually is usually poped in local ground
        // frame, not the global ground frame.
        if (all_offline_object_cubes.size() == 0)
            return;
        pop_pose_to_ground = InitToGround.clone(); // for kitti, I used InitToGround to pop offline.

        Eigen::MatrixXd pred_frame_objects;
        Eigen::MatrixXd pred_truth_matches;

        pred_frame_objects = all_offline_object_cubes[(int)pKF->mnFrameId];

        for (int i = 0; i < pred_frame_objects.rows(); i++)
        {
            cuboid *raw_cuboid = new cuboid();
            raw_cuboid->pos = pred_frame_objects.row(i).head(3);
            raw_cuboid->rotY = pred_frame_objects(i, 3);
            raw_cuboid->scale = Vector3d(pred_frame_objects(i, 4), pred_frame_objects(i, 5),
                                         pred_frame_objects(i, 6));
            raw_cuboid->rect_detect_2d = pred_frame_objects.row(i).segment<4>(7);
            raw_cuboid->box_config_type =
                Vector2d(1, 1); // randomly given unless provided. for latter visualization
            all_obj2d_bbox.push_back(raw_cuboid->rect_detect_2d);
            all_box_confidence.push_back(pred_frame_objects(i, 11));
            if (use_truth_trackid)
                truth_tracklet_ids.push_back(pred_frame_objects(i, 12));
            ObjectSet temp;
            temp.push_back(raw_cuboid);
            all_obj_cubes.push_back(temp);
        }
    }
    else
    {
        pop_pose_to_ground = InitToGround.clone();

        std::string data_yolo_obj_dir = data_folder + "/bb_tracking/";
        char frame_index_c[256];
        sprintf(frame_index_c, "%06d", (int)pKF->mnFrameId); // format into 4 digit

        // read detected edges
        // Eigen::MatrixXd all_lines_raw(
        //     100, 4); // 100 is some large frame number,   the txt edge index start from 0
        // read_all_number_txt(data_edge_data_dir + frame_index_c + "_edge.txt", all_lines_raw);

        // read yolo object detection
        Eigen::MatrixXd raw_all_obj2d_bbox(10, 6);
        std::vector<string> object_classes;
        if (!read_obj_detection_txt(data_yolo_obj_dir + frame_index_c + ".txt", raw_all_obj2d_bbox,
                                    object_classes))
        {
            ROS_WARN_STREAM("Cannot read object detection BB for frame "
                            << frame_index_c << ", assuming no detected BB");
        }
        else
        {
            // remove some 2d boxes too close to boundary.
            int boundary_threshold = 20;
            int img_width = pKF->raw_img.cols;
            std::vector<int> good_object_ids;
            for (int i = 0; i < raw_all_obj2d_bbox.rows(); i++)
                if ((raw_all_obj2d_bbox(i, 0) > boundary_threshold) &&
                    (raw_all_obj2d_bbox(i, 0) + raw_all_obj2d_bbox(i, 2) <
                     img_width - boundary_threshold))
                    good_object_ids.push_back(i);
            Eigen::MatrixXd all_obj2d_bbox_infov_mat(good_object_ids.size(), 6);
            for (size_t i = 0; i < good_object_ids.size(); i++)
            {
                all_obj2d_bbox_infov_mat.row(i) = raw_all_obj2d_bbox.row(good_object_ids[i]);
                all_obj2d_bbox.push_back(raw_all_obj2d_bbox.row(good_object_ids[i]));
                all_box_confidence.push_back(raw_all_obj2d_bbox(good_object_ids[i], 4));
            }

            cv::Mat frame_pose_to_init = pKF->GetPoseInverse(); // camera to init world
            cv::Mat frame_pose_to_ground = frame_pose_to_init;  // to my ground frame
            if (!build_worldframe_on_ground)
                frame_pose_to_ground = InitToGround * frame_pose_to_ground;

            pop_pose_to_ground = InitToGround.clone();
            Eigen::Matrix4f cam_transToGround = Converter::toMatrix4f(pop_pose_to_ground);
            detect_cuboid_obj->detect_cuboid(pKF->raw_img, cam_transToGround.cast<double>(),
                                             all_obj2d_bbox_infov_mat, all_obj_cubes);
        }
    }

    // copy and analyze results. change to g2o cuboid.
    pKF->local_cuboids.clear();
    g2o::SE3Quat frame_pose_to_init =
        Converter::toSE3Quat(pKF->GetPoseInverse()); // camera to init, not always ground.
    g2o::SE3Quat InitToGround_se3 = Converter::toSE3Quat(InitToGround);
    for (int ii = 0; ii < (int)all_obj_cubes.size(); ii++)
    {
        if (all_obj_cubes[ii].size() > 0) // if has detected 3d Cuboid
        {
            cuboid *raw_cuboid = all_obj_cubes[ii][0];

            g2o::cuboid
                cube_ground_value; // offline cuboid txt in local ground frame.  [x y z yaw l w h]
            Vector9d cube_pose;
            cube_pose << raw_cuboid->pos[0], raw_cuboid->pos[1], raw_cuboid->pos[2], 0, 0,
                raw_cuboid->rotY, raw_cuboid->scale[0], raw_cuboid->scale[1], raw_cuboid->scale[2];
            cube_ground_value.fromMinimalVector(cube_pose);

            // measurement in local camera frame! important
            MapObject *newcuboid = new MapObject(mpMap);
            g2o::cuboid cube_local_meas =
                cube_ground_value.transform_to(Converter::toSE3Quat(pop_pose_to_ground));
            newcuboid->cube_meas = cube_local_meas;
            newcuboid->bbox_2d =
                cv::Rect(raw_cuboid->rect_detect_2d[0], raw_cuboid->rect_detect_2d[1],
                         raw_cuboid->rect_detect_2d[2], raw_cuboid->rect_detect_2d[3]);
            newcuboid->bbox_vec =
                Vector4d((double)newcuboid->bbox_2d.x + (double)newcuboid->bbox_2d.width / 2,
                         (double)newcuboid->bbox_2d.y + (double)newcuboid->bbox_2d.height / 2,
                         (double)newcuboid->bbox_2d.width, (double)newcuboid->bbox_2d.height);
            newcuboid->box_corners_2d = raw_cuboid->box_corners_2d;
            newcuboid->bbox_2d_tight =
                cv::Rect(raw_cuboid->rect_detect_2d[0] + raw_cuboid->rect_detect_2d[2] / 10.0,
                         raw_cuboid->rect_detect_2d[1] + raw_cuboid->rect_detect_2d[3] / 10.0,
                         raw_cuboid->rect_detect_2d[2] * 0.8, raw_cuboid->rect_detect_2d[3] * 0.8);
            get_cuboid_draw_edge_markers(newcuboid->edge_markers, raw_cuboid->box_config_type,
                                         false);
            newcuboid->SetReferenceKeyFrame(pKF);
            newcuboid->object_id_in_localKF = pKF->local_cuboids.size();

            g2o::cuboid global_obj_pose_to_init =
                cube_local_meas.transform_from(frame_pose_to_init);

            newcuboid->SetWorldPos(global_obj_pose_to_init);
            newcuboid->pose_noopti = global_obj_pose_to_init;
            if (use_truth_trackid)
            {
                if (whether_read_offline_cuboidtxt)
                    newcuboid->truth_tracklet_id = truth_tracklet_ids[ii];
                else
                    newcuboid->truth_tracklet_id = raw_cuboid->track_2d_id;
            }

            if (whether_dynamic_object)
            {
                newcuboid->is_dynamic = true; // for debug, later should check!
                newcuboid->pose_Twc_latestKF =
                    global_obj_pose_to_init; // set pose for dynamic object
            }

            if (scene_unique_id == kitti)
            {
                if (cube_local_meas.pose.translation()(0) > 1)
                    newcuboid->left_right_to_car = 2; // right
                if (cube_local_meas.pose.translation()(0) < -1)
                    newcuboid->left_right_to_car = 1; // left
                if ((cube_local_meas.pose.translation()(0) > -1) &&
                    (cube_local_meas.pose.translation()(0) < 1))
                    newcuboid->left_right_to_car = 0;
            }
            if (1)
            {
                double obj_cam_dist = std::min(
                    std::max(newcuboid->cube_meas.translation()(2), 10.0), 30.0); // cut into [a,b]
                double obj_meas_quality = (60.0 - obj_cam_dist) / 40.0;
                newcuboid->meas_quality = obj_meas_quality;
            }
            else
                newcuboid->meas_quality = 1.0;
            if (all_box_confidence[ii] > 0)
                newcuboid->meas_quality *= all_box_confidence[ii]; // or =

            if (newcuboid->meas_quality < 0.1)
                ROS_WARN_STREAM("Abnormal measure quality!!:   " << newcuboid->meas_quality);
            pKF->local_cuboids.push_back(newcuboid);
        }
    }

    std::cout << "created local object num   " << pKF->local_cuboids.size() << std::endl;
    std::cout << "Detect cuboid for pKF id: " << pKF->mnId << "  total id: " << pKF->mnFrameId
              << "  numObj: " << pKF->local_cuboids.size() << std::endl;

    if (whether_save_online_detected_cuboids)
    {
        for (int ii = 0; ii < (int)all_obj_cubes.size(); ii++)
        {
            if (all_obj_cubes[ii].size() > 0) // if has detected 3d Cuboid, always true in this case
            {
                cuboid *raw_cuboid = all_obj_cubes[ii][0];
                g2o::cuboid cube_ground_value;
                Vector9d cube_pose;
                cube_pose << raw_cuboid->pos[0], raw_cuboid->pos[1], raw_cuboid->pos[2], 0, 0,
                    raw_cuboid->rotY, raw_cuboid->scale[0], raw_cuboid->scale[1],
                    raw_cuboid->scale[2];
                save_online_detected_cuboids << pKF->mnFrameId << "  " << cube_pose.transpose()
                                             << "\n";
            }
        }
    }

    if (associate_point_with_object)
    {
        if (!whether_dynamic_object) // for old non-dynamic object, associate based on 2d overlap...
                                     // could also use instance segmentation
        {
            pKF->keypoint_associate_objectID = vector<int>(pKF->mvKeys.size(), -1);
            std::vector<bool> overlapped(pKF->local_cuboids.size(), false);
            if (1)
            {
                for (size_t i = 0; i < pKF->local_cuboids.size(); i++)
                    if (!overlapped[i])
                        for (size_t j = i + 1; j < pKF->local_cuboids.size(); j++)
                            if (!overlapped[j])
                            {
                                float iou_ratio = bboxOverlapratio(pKF->local_cuboids[i]->bbox_2d,
                                                                   pKF->local_cuboids[j]->bbox_2d);
                                if (iou_ratio > 0.15)
                                {
                                    overlapped[i] = true;
                                    overlapped[j] = true;
                                }
                            }
            }

            if (!enable_ground_height_scale)
            {                                      // slightly faster
                if (pKF->local_cuboids.size() > 0) // if there is object
                    for (size_t i = 0; i < pKF->mvKeys.size(); i++)
                    {
                        int associated_times = 0;
                        for (size_t j = 0; j < pKF->local_cuboids.size(); j++)
                            if (!overlapped[j])
                                if (pKF->local_cuboids[j]->bbox_2d.contains(pKF->mvKeys[i].pt))
                                {
                                    associated_times++;
                                    if (associated_times == 1)
                                        pKF->keypoint_associate_objectID[i] = j;
                                    else
                                        pKF->keypoint_associate_objectID[i] = -1;
                                }
                    }
            }
            else
            {
                pKF->keypoint_inany_object = vector<bool>(pKF->mvKeys.size(), false);
                for (size_t i = 0; i < pKF->mvKeys.size(); i++)
                {
                    int associated_times = 0;
                    for (size_t j = 0; j < pKF->local_cuboids.size(); j++)
                        if (pKF->local_cuboids[j]->bbox_2d.contains(pKF->mvKeys[i].pt))
                        {
                            pKF->keypoint_inany_object[i] = true;
                            if (!overlapped[j])
                            {
                                associated_times++;
                                if (associated_times == 1)
                                    pKF->keypoint_associate_objectID[i] = j;
                                else
                                    pKF->keypoint_associate_objectID[i] = -1;
                            }
                        }
                }
                if (height_esti_history.size() == 0)
                {
                    pKF->local_cuboids.clear(); // don't do object when in initial stage...
                    pKF->keypoint_associate_objectID.clear();
                }
            }
        }

        if (whether_dynamic_object) //  for dynamic object, I use instance segmentation
        {
            if (pKF->local_cuboids.size() > 0) // if there is object
            {
                std::vector<MapPoint *> framePointMatches = pKF->GetMapPointMatches();

                if (pKF->keypoint_associate_objectID.size() < pKF->mvKeys.size())
                    ROS_ERROR_STREAM("Tracking Bad keypoint associate ID size   "
                                     << pKF->keypoint_associate_objectID.size() << "  "
                                     << pKF->mvKeys.size());

                for (size_t i = 0; i < pKF->mvKeys.size(); i++)
                {
                    if (pKF->keypoint_associate_objectID[i] >= 0 &&
                        pKF->keypoint_associate_objectID[i] >= pKF->local_cuboids.size())
                    {
                        ROS_ERROR_STREAM("Detect cuboid find bad pixel obj id  "
                                         << pKF->keypoint_associate_objectID[i] << "  "
                                         << pKF->local_cuboids.size());
                    }
                    if (pKF->keypoint_associate_objectID[i] > -1)
                    {
                        MapPoint *pMP = framePointMatches[i];
                        if (pMP)
                            pMP->is_dynamic = true;
                    }
                }
            }
        }
    }

    std::vector<KeyFrame *> checkframes = mvpLocalKeyFrames; // only check recent to save time

    int object_own_point_threshold = 20;
    if (scene_unique_id == kitti)
    {
        if (mono_allframe_Obj_depth_init)
            object_own_point_threshold =
                50; // 50 using 10 is too noisy.... many objects don't have enough points to match
                    // with others then will create as new...
        else
            object_own_point_threshold =
                30; // 30 if don't initialize object point sepratedly, there won't be many
                    // points....  tried 20, not good...
    }

    if (whether_dynamic_object)
    {
        if (mono_allframe_Obj_depth_init)
            object_own_point_threshold = 10;
    }

    if (use_truth_trackid) // very accurate, no need of object point for association
        object_own_point_threshold = -1;

    // points and object are related in local mapping, when creating mapPoints

    // dynamic object: didn't triangulate point in localmapping. but in tracking
    for (size_t i = 0; i < checkframes.size(); i++)
    {
        KeyFrame *kfs = checkframes[i];
        for (size_t j = 0; j < kfs->local_cuboids.size(); j++)
        {
            MapObject *mPO = kfs->local_cuboids[j];
            if (!mPO->become_candidate)
            {
                // points number maybe increased when later triangulated
                mPO->check_whether_valid_object(object_own_point_threshold);
            }
        }
    }
}

} // namespace ORB_SLAM2
