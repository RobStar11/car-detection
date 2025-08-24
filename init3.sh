#!/bin/bash

# --- Automatic FFmpeg Installation for Raspberry Pi OS (Debian-based) ---
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg not found. Attempting to install it now..."
    # Update package lists and install ffmpeg without confirmation prompts
    sudo apt-get update
    sudo apt-get install -y ffmpeg
    
    # Verify the installation was successful
    if ! command -v ffmpeg &> /dev/null
    then
        echo "ERROR: ffmpeg installation failed. Please try installing it manually."
        exit 1
    fi
    echo "ffmpeg has been successfully installed."
fi
# --- End of Installation Logic ---


MEDIAMTX_DIR="mediamtx"
MEDIAMTX_VERSION="v1.14.0" # You can update this version if needed

# Create the mediamtx directory if it doesn't exist and enter it
mkdir -p "$MEDIAMTX_DIR"
cd "$MEDIAMTX_DIR"

# Check if the mediamtx executable exists. If not, download the correct version.
if [ ! -f "./mediamtx" ]; then
    echo "mediamtx executable not found. Detecting Raspberry Pi architecture..."

    # Detect the system architecture using uname
    ARCH=$(uname -m)
    MEDIAMTX_ARCH_SUFFIX=""

    # Determine the correct download suffix based on the architecture
    case "$ARCH" in
        aarch64 | arm64)
            echo "Detected 64-bit ARM (aarch64). Downloading arm64 binary."
            MEDIAMTX_ARCH_SUFFIX="arm64"
            ;;
        armv7l)
            echo "Detected 32-bit ARM (armv7l). Downloading armv7 binary."
            MEDIAMTX_ARCH_SUFFIX="armv7"
            ;;
        *)
            echo "Error: Unsupported Raspberry Pi architecture: $ARCH"
            echo "This script supports arm64 and armv7. Please download mediamtx manually."
            exit 1
            ;;
    esac

    # Construct the download URL and filename
    FILENAME="mediamtx_${MEDIAMTX_VERSION}_linux_${MEDIAMTX_ARCH_SUFFIX}.tar.gz"
    DOWNLOAD_URL="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/${FILENAME}"

    echo "Downloading from: $DOWNLOAD_URL"
    wget "$DOWNLOAD_URL"
    
    # Extract the downloaded archive
    tar -xvzf "$FILENAME"
fi

# Function to clean up the background server on exit
cleanup() {
    echo -e "\nShutting down ffmpeg and mediamtx server..."
    # Find and kill the mediamtx process that this script started
    kill $(jobs -p) 2>/dev/null
    exit
}

# Trap Ctrl+C and other exit signals to run the cleanup function
trap cleanup SIGINT SIGTERM

# Start the mediamtx server in the background
echo "Starting mediamtx server..."
./mediamtx &

# Give the server a moment to initialize
sleep 2

# Start the ffmpeg stream
echo "Starting ffmpeg stream... (Press Ctrl+C to stop)"
ffmpeg -hide_banner -f v4l2 -framerate 15 -video_size 640x480 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2M -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1

# After ffmpeg stops, run the cleanup function to stop mediamtx
cleanup
