cmake_minimum_required(VERSION 3.10)
project(HeadPoseNCNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ncnn_DIR /home/chikara/hardware/work/build_ncnn/ncnn/build/install/lib/cmake/ncnn)

# 最適化フラグ（PCのCPU性能を出す設定）
add_compile_options(-O3 -march=native)

# 依存パッケージの検索
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
find_package(realsense2 REQUIRED)

# インクルードディレクトリの設定
include_directories(${OpenCV_INCLUDE_DIRS})

# 実行ファイルの作成（メインのC++ファイルを指定）
add_executable(HeadPose realsense.cpp FaceDetector.cpp)

# ライブラリのリンク
target_link_libraries(HeadPose ${OpenCV_LIBS} ncnn realsense2)
