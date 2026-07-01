#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <vector>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "FaceDetector.h"

using namespace std;

struct FaceData {
    cv::Point2d nose_tip;
    float dist_m;
    double pitch;
    cv::Mat rvec, tvec;
};

int main(int argc, char **argv)
{
    float f;
    float FPS[16];
    int i, Fcnt=0;
    cv::Mat frame;
    const int max_side = 320;

    // ---- RealSense setup ----
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16);
    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::align align_to(RS2_STREAM_COLOR);

    rs2_intrinsics intr = profile.get_stream(RS2_STREAM_COLOR)
                                 .as<rs2::video_stream_profile>().get_intrinsics();

    chrono::steady_clock::time_point Tbegin, Tend;
    for(i=0;i<16;i++) FPS[i]=0.0;

    // ---- Camera matrix ----
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3)
        << intr.fx, 0,       intr.ppx,
           0,       intr.fy, intr.ppy,
           0,       0,       1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

    // ---- 3D face model (5-point, no chin) ----
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(   0.0f,    0.0f,    0.0f));  // Nose tip
    model_points.push_back(cv::Point3d(-225.0f,  170.0f, -135.0f));  // Left eye
    model_points.push_back(cv::Point3d( 225.0f,  170.0f, -135.0f));  // Right eye
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));  // Left mouth
    model_points.push_back(cv::Point3d( 150.0f, -150.0f, -125.0f));  // Right mouth

    // ---- Axis points ----
    vector<cv::Point3d> axis_pts3D;
    axis_pts3D.push_back(cv::Point3d(300.0, 0.0,   0.0));
    axis_pts3D.push_back(cv::Point3d(0.0, 300.0,   0.0));
    axis_pts3D.push_back(cv::Point3d(0.0,   0.0, 300.0));

    // Per-speaker label colors (BGR): speaker1=cyan, speaker2=orange, speaker3=magenta
    cv::Scalar speaker_colors[3] = {
        cv::Scalar(255, 255,   0),
        cv::Scalar(  0, 165, 255),
        cv::Scalar(255,   0, 255)
    };
    const char* speaker_names[3] = {"speaker1", "speaker2", "speaker3"};

    Detector detector("face.param", "face.bin");

    // ---- Pitch calibration ----
    // c: look forward -> sets zero
    // u: look straight up -> sets 90 deg
    double pitch_offset   = 0.0;
    double pitch_scale    = 1.5;
    double last_pitch_raw = 0.0;

    // ---- Log file (CSV, append) ----
    std::ofstream log_file("output.log", std::ios::app);
    log_file << "speaker1_dist,speaker1_pitch,speaker2_dist,speaker2_pitch,"
                "speaker3_dist,speaker3_pitch\n";

    while(true){
        Tbegin = chrono::steady_clock::now();

        // ---- Grab frames ----
        rs2::frameset frames        = pipe.wait_for_frames();
        rs2::frameset aligned       = align_to.process(frames);
        rs2::video_frame color_frame = aligned.get_color_frame();
        rs2::depth_frame depth_frame = aligned.get_depth_frame();
        rs2_intrinsics depth_intr   = depth_frame.get_profile()
                                        .as<rs2::video_stream_profile>().get_intrinsics();

        frame = cv::Mat(cv::Size(640, 480), CV_8UC3,
                        (void*)color_frame.get_data(), cv::Mat::AUTO_STEP).clone();

        // ---- Face detection on scaled image ----
        float long_side = std::max(frame.cols, frame.rows);
        float scale     = max_side / long_side;
        cv::Mat img_scale;
        cv::resize(frame, img_scale, cv::Size(frame.cols*scale, frame.rows*scale));

        std::vector<bbox> boxes;
        detector.Detect(img_scale, boxes);

        Tend = chrono::steady_clock::now();

        // ---- Compute distance + pitch for every detected face ----
        std::vector<FaceData> all_faces;

        for (size_t j = 0; j < boxes.size(); ++j) {
            std::vector<cv::Point2d> image_points;
            image_points.push_back(cv::Point2d(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale));
            image_points.push_back(cv::Point2d(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale));
            image_points.push_back(cv::Point2d(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale));
            image_points.push_back(cv::Point2d(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale));
            image_points.push_back(cv::Point2d(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale));

            cv::Mat rvec, tvec, R;
            cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                         rvec, tvec, false, cv::SOLVEPNP_SQPNP);
            cv::Rodrigues(rvec, R);
            cv::Mat flip = (cv::Mat_<double>(3,3) << 1,0,0, 0,1,0, 0,0,-1);
            R = flip * R;

            double pitch_rad = atan2(R.at<double>(2,1), R.at<double>(2,2));
            double pitch_raw = -(pitch_rad * 180.0 / CV_PI);
            double pitch = (pitch_raw - pitch_offset) * pitch_scale;
            if (pitch >  90.0) pitch -= 180.0;
            if (pitch < -90.0) pitch += 180.0;

            cv::Point2d nose_tip = image_points[0];
            int dx = std::clamp((int)nose_tip.x, 0, 639);
            int dy = std::clamp((int)nose_tip.y, 0, 479);
            float pixel[2] = {(float)dx, (float)dy};
            float pt3d[3];
            rs2_deproject_pixel_to_point(pt3d, &depth_intr, pixel,
                                         depth_frame.get_distance(dx, dy));
            float dist_m = std::sqrt(pt3d[0]*pt3d[0] + pt3d[1]*pt3d[1] + pt3d[2]*pt3d[2]);

            FaceData fd;
            fd.nose_tip = nose_tip;
            fd.dist_m   = dist_m;
            fd.pitch    = pitch;
            fd.rvec     = rvec.clone();
            fd.tvec     = tvec.clone();
            all_faces.push_back(fd);

            if (j == 0) last_pitch_raw = pitch_raw;
        }

        // ---- Sort by distance (closest first), keep top 3 ----
        std::sort(all_faces.begin(), all_faces.end(),
                  [](const FaceData& a, const FaceData& b){ return a.dist_m < b.dist_m; });
        if (all_faces.size() > 3) all_faces.resize(3);

        // ---- Draw each speaker ----
        for (size_t k = 0; k < all_faces.size(); ++k) {
            const FaceData& fd = all_faces[k];
            cv::Scalar color   = speaker_colors[k];
            const char* label  = speaker_names[k];

            vector<cv::Point2d> axis_pts2D;
            cv::projectPoints(axis_pts3D, fd.rvec, fd.tvec, camera_matrix, dist_coeffs, axis_pts2D);
            cv::line(frame, fd.nose_tip, axis_pts2D[0], cv::Scalar(0,   0, 255), 2);
            cv::line(frame, fd.nose_tip, axis_pts2D[1], cv::Scalar(0, 255,   0), 2);
            cv::line(frame, fd.nose_tip, axis_pts2D[2], cv::Scalar(255,   0, 0), 2);

            int tx = std::clamp((int)fd.nose_tip.x + 10, 0, 580);
            int ty = std::clamp((int)fd.nose_tip.y - 40, 20, 460);

            std::ostringstream ss_dist, ss_pitch;
            ss_dist  << std::fixed << std::setprecision(2) << fd.dist_m << " m";
            ss_pitch << std::fixed << std::setprecision(1) << fd.pitch  << " deg";

            cv::putText(frame, label,                       cv::Point(tx, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2);
            cv::putText(frame, "Dist:  " + ss_dist.str(),  cv::Point(tx, ty + 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.60, color, 2);
            cv::putText(frame, "Pitch: " + ss_pitch.str(), cv::Point(tx, ty + 44),
                        cv::FONT_HERSHEY_SIMPLEX, 0.60, color, 2);
        }

        // ---- Write one CSV line per frame ----
        for (int k = 0; k < 3; ++k) {
            if (k > 0) log_file << ",";
            if (k < (int)all_faces.size() && all_faces[k].dist_m > 0.0f) {
                log_file << std::fixed << std::setprecision(3) << all_faces[k].dist_m
                         << ","
                         << std::fixed << std::setprecision(1) << all_faces[k].pitch;
            } else {
                log_file << "nan,nan";
            }
        }
        log_file << "\n";
        log_file.flush();

        // ---- FPS (top-left) ----
        f = chrono::duration_cast<chrono::milliseconds>(Tend - Tbegin).count();
        if(f > 0.0) FPS[((Fcnt++)&0x0F)] = 1000.0f / f;
        for(f=0.0, i=0; i<16; i++) f += FPS[i];
        cv::putText(frame, cv::format("FPS: %.1f", f/16),
                    cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        cv::imshow("Head Pose", frame);
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'c') {
            pitch_offset = last_pitch_raw;
            std::cout << "Forward calibrated: offset=" << pitch_offset << " deg" << std::endl;
        }
        if (key == 'u') {
            double diff = last_pitch_raw - pitch_offset;
            if (std::abs(diff) > 1.0) {
                pitch_scale = 90.0 / diff;
                std::cout << "Up calibrated: scale=" << pitch_scale << std::endl;
            }
        }
    }

    log_file.close();
    cv::destroyAllWindows();
    return 0;
}
