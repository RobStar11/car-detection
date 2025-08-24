#!/bin/bash

# The directory where mediamtx will be run
MEDIAMTX_DIR="mediamtx"

# The full path to the mediamtx executable
MEDIAMTX_EXECUTABLE="./${MEDIAMTX_DIR}/mediamtx"

# Create the directory if it doesn't exist
mkdir -p $MEDIAMTX_DIR
cd $MEDIAMTX_DIR

# Check if the mediamtx executable exists. [2, 4]
if [ ! -f "./mediamtx" ]; then
    echo "mediamtx executable not found. Downloading and extracting..."
    # Download the specified version of mediamtx
    wget https://github.com/bluenviron/mediamtx/releases/download/v1.14.0/mediamtx_v1.14.0_linux_arm64.tar.gz
    
    # Extract the contents of the tarball
    tar -xvzf mediamtx_v1.14.0_linux_arm64.tar.gz
fi

# Start the mediamtx server in the background
echo "Starting mediamtx server..."
./mediamtx &

# Give the server a moment to initialize
sleep 2

# Start the ffmpeg stream
echo "Starting ffmpeg stream..."
ffmpeg -hide_banner -f v4l2 -framerate 15 -video_size 640x480 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2M -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1
