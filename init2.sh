#!/bin/bash

# --- Automatic FFmpeg Installation for Ubuntu ---
# Check if the 'ffmpeg' command is available in the system's PATH.
if ! command -v ffmpeg &> /dev/null
then
    echo "-----------------------------------------------------"
    echo "ffmpeg is not installed. Attempting to install it now..."
    echo "You may be asked to enter your password for the installation."
    echo "-----------------------------------------------------"
    
    # First, update the package lists to ensure we get the correct version
    sudo apt-get update
    
    # Now, install ffmpeg. The '-y' flag automatically confirms the installation.
    sudo apt-get install -y ffmpeg
    
    # After attempting installation, check again to make sure it was successful.
    if ! command -v ffmpeg &> /dev/null
    then
        echo "-----------------------------------------------------"
        echo "ERROR: ffmpeg installation failed."
        echo "Please try installing it manually by running:"
        echo "sudo apt update && sudo apt install ffmpeg"
        echo "-----------------------------------------------------"
        exit 1 # Exit the script if installation fails
    else
        echo "ffmpeg has been successfully installed."
    fi
fi
# --- End of Installation Logic ---


# Create the mediamtx directory if it doesn't exist and enter it
mkdir -p mediamtx
cd mediamtx

# NOTE: The "Exec format error" you saw before means this 'arm64' version
# is likely incorrect for your system. If the error persists, you should
# replace 'linux_arm64' in the URL below with 'linux_amd64'.
if [ ! -f "./mediamtx" ]; then
    echo "mediamtx not found. Downloading..."
    wget https://github.com/bluenviron/mediamtx/releases/download/v1.14.0/mediamtx_v1.14.0_linux_arm64.tar.gz
    tar -xvzf mediamtx_v1.14.0_linux_arm64.tar.gz
fi

# Function to clean up the background server on exit
cleanup() {
    echo "Shutting down ffmpeg and mediamtx server..."
    kill $MEDIAMTX_PID
    exit
}

# Trap Ctrl+C and other exit signals to run the cleanup function
trap cleanup SIGINT SIGTERM

# Start the mediamtx server in the background
echo "Starting mediamtx server..."
./mediamtx &
# Store the Process ID (PID) of the mediamtx server so we can stop it later
MEDIAMTX_PID=$!

# Give the server a moment to initialize
sleep 2

# Start the ffmpeg stream
echo "Starting ffmpeg stream... (Press Ctrl+C to stop)"
ffmpeg -hide_banner -f v4l2 -framerate 15 -video_size 640x480 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2M -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1

# After ffmpeg stops, run the cleanup function to stop mediamtx
cleanup```

### How to Use

1.  **Save the Script:** Save the code above to a file named `start_stream.sh`.
2.  **Make it Executable:**
    ```bash
    chmod +x start_stream.sh
    ```
3.  **Run the Script:**
    ```bash
    ./start_stream.sh
    ```
When you run the script, if `ffmpeg` is missing, it will begin the installation. Simply enter your password when prompted and let it complete. The script will then continue to start the servers and your stream.
