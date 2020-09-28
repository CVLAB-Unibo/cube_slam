/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include "multi_settings/cli_settings.h"
#include "multi_settings/structured_settings.h"

#include "orb_object_slam/Parameters.h"
#include "orb_object_slam/System.h"

void loadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        std::getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/images/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "Usage: cube_slam SCENE_PATH DATA_PATH ORB_VOCABULARY_PATH MAIN_SETTINGS_PATH "
                  << "[OTH_SETTINGS_PATH] [OPTION=VALUE ...]"
                  << "\n";
        std::exit(1);
    }

    ORB_SLAM2::scene_folder = argv[1];
    ORB_SLAM2::data_folder = argv[2];
    std::string voc_path = argv[3];

    ORB_SLAM2::otherSettings.add(settings::StructuredSettings(argv[4]));
    if (argc >= 6)
    {
        int cliSettingsStart = 5;
        struct stat buffer;
        if (stat(argv[cliSettingsStart], &buffer) == 0)
        {
            ORB_SLAM2::otherSettings.add(settings::StructuredSettings(argv[cliSettingsStart]));
            cliSettingsStart++;
        }

        if (cliSettingsStart < argc)
        {
            std::cout << "Using " << (argc - cliSettingsStart) << " settings from CLI\n";
            ORB_SLAM2::otherSettings.add(settings::CLISettings(
                std::vector<std::string>(argv + cliSettingsStart, argv + argc)));
        }
    }

    std::cout << "use_truth_trackid: " << (bool)ORB_SLAM2::otherSettings["use_truth_trackid"]
              << "\n\n";

    std::string scene_name = ORB_SLAM2::otherSettings["scene_name"];

    ORB_SLAM2::enable_viewer = ORB_SLAM2::otherSettings["enable_viewer"];
    ORB_SLAM2::enable_viewmap = ORB_SLAM2::otherSettings["enable_viewmap"];
    ORB_SLAM2::enable_viewimage = ORB_SLAM2::otherSettings["enable_viewimage"];
    bool enable_loop_closing = ORB_SLAM2::otherSettings["enable_loop_closing"];
    ORB_SLAM2::parallel_mapping = ORB_SLAM2::otherSettings["parallel_mapping"];

    ORB_SLAM2::whether_detect_object = ORB_SLAM2::otherSettings["whether_detect_object"];
    ORB_SLAM2::whether_read_offline_cuboidtxt =
        ORB_SLAM2::otherSettings["whether_read_offline_cuboidtxt"];
    ORB_SLAM2::associate_point_with_object =
        ORB_SLAM2::otherSettings["associate_point_with_object"];

    ORB_SLAM2::whether_dynamic_object = ORB_SLAM2::otherSettings["whether_dynamic_object"];
    ORB_SLAM2::remove_dynamic_features = ORB_SLAM2::otherSettings["remove_dynamic_features"];

    ORB_SLAM2::mono_firstframe_truth_depth_init =
        ORB_SLAM2::otherSettings["mono_firstframe_truth_depth_init"];
    ORB_SLAM2::mono_firstframe_Obj_depth_init =
        ORB_SLAM2::otherSettings["mono_firstframe_Obj_depth_init"];
    ORB_SLAM2::mono_allframe_Obj_depth_init =
        ORB_SLAM2::otherSettings["mono_allframe_Obj_depth_init"];

    ORB_SLAM2::enable_ground_height_scale = ORB_SLAM2::otherSettings["enable_ground_height_scale"];
    ORB_SLAM2::use_dynamic_klt_features = ORB_SLAM2::otherSettings["use_dynamic_klt_features"];

    ORB_SLAM2::bundle_object_opti = ORB_SLAM2::otherSettings["bundle_object_opti"];
    ORB_SLAM2::camera_object_BA_weight = ORB_SLAM2::otherSettings["camera_object_BA_weight"];
    ORB_SLAM2::object_velocity_BA_weight = ORB_SLAM2::otherSettings["object_velocity_BA_weight"];

    ORB_SLAM2::draw_map_truth_paths = ORB_SLAM2::otherSettings["draw_map_truth_paths"];
    ORB_SLAM2::draw_nonlocal_mappoint = ORB_SLAM2::otherSettings["draw_nonlocal_mappoint"];

    // temp debug
    ORB_SLAM2::ba_dyna_pt_obj_cam = ORB_SLAM2::otherSettings["ba_dyna_pt_obj_cam"];
    ORB_SLAM2::ba_dyna_obj_velo = ORB_SLAM2::otherSettings["ba_dyna_obj_velo"];
    ORB_SLAM2::ba_dyna_obj_cam = ORB_SLAM2::otherSettings["ba_dyna_obj_cam"];

    if (scene_name.compare(std::string("kitti")) == 0)
        ORB_SLAM2::scene_unique_id = ORB_SLAM2::kitti;

    if (!enable_loop_closing)
        std::cout << "Turn off global loop closing!!";
    else
        std::cout << "Turn on global loop closing!!";

    std::string mode_name = ORB_SLAM2::otherSettings["sensor_mode"];
    ORB_SLAM2::System::eSensor mode;
    if (mode_name == "mono")
    {
        mode = ORB_SLAM2::System::MONOCULAR;
    }
    else if (mode_name == "stereo")
    {
        mode = ORB_SLAM2::System::STEREO;
    }
    else if (mode_name == "rgbd")
    {
        mode = ORB_SLAM2::System::RGBD;
    }
    else
    {
        std::cerr << "Unknown sensor mode '" << mode_name << "'\n";
        std::exit(2);
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(voc_path, mode, enable_loop_closing);

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    loadImages(ORB_SLAM2::scene_folder, vstrImageFilenames, vTimestamps);

    cv::Mat im;
    const auto nImages = vstrImageFilenames.size();
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe, ni);

        // if (ni < nImages)
        //     usleep(0.1 * 1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    //     SLAM.SaveKeyFrameTrajectoryTUM(packagePath+"/Outputs/KeyFrameTrajectory.txt");

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM(ORB_SLAM2::data_folder + "/AllFrameTrajectory.txt");
    if (ORB_SLAM2::scene_unique_id == ORB_SLAM2::kitti)
        SLAM.SaveTrajectoriesKITTI(ORB_SLAM2::data_folder + "/AllFrameTrajectoryKITTI.txt");

    return 0;
}
