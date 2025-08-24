# To know the raspberry ip
hostname -I

# Download the server
wget https://github.com/bluenviron/mediamtx/releases/download/v1.14.0/mediamtx_v1.14.0_linux_arm64.tar.gz
tar -xvzf mediamtx_v1.14.0_linux_arm64.tar.gz

# To expose the server
cd mediamtx
./mediamtx

# To expose cam1
ffmpeg -hide_banner   -f v4l2 -framerate 15 -video_size 640x480 -i /dev/video0   -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2M -pix_fmt yuv420p   -f rtsp -rtsp_transport tcp   rtsp://127.0.0.1:8554/cam1

