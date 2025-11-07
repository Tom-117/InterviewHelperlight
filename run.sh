#!/usr/bin/bash
set -e

DOCKER_IMAGE="tom0117/interview-helper-lite:webui"
GPU_FLAG=""

# Check if  GPU is available
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo "NVIDIA GPU detected - GPU acceleration will be enabled"
fi

print_menu() {
    cat <<EOF

╔════════════════════════════════════════════════════╗
║           Interview Helper Light                   ║
╚════════════════════════════════════════════════════╝

    1) Start Web Interface    -> http://localhost:5000
    2) Start CLI Version
    0) Exit

    Models Directory: $(pwd)/models
    Uploads Directory: $(pwd)/uploads
EOF

    # Show GPU status
    if [ -n "$GPU_FLAG" ]; then
        echo "    GPU: Enabled ($(nvidia-smi --query-gpu=name --format=csv,noheader))"
    else
        echo "    GPU: Not available (CPU mode)"
    fi
    echo
}

ensure_dirs() {
    mkdir -p "$(pwd)/models" "$(pwd)/uploads"
    chmod -R 777 "$(pwd)/models" "$(pwd)/uploads" 2>/dev/null || true
}

# Main loop
while true; do
    clear
    print_menu
    read -rp "Choose [0-2]: " choice
    echo

    case "$choice" in
        1)
            ensure_dirs
            echo "Starting web interface..."
            
            # Only pull if image doesn't exist locally
            if ! docker image inspect $DOCKER_IMAGE >/dev/null 2>&1; then
                echo "Image not found locally, pulling from Docker Hub..."
                docker pull $DOCKER_IMAGE
            fi

            docker run -it --rm \
                $GPU_FLAG \
                -p 5000:5000 \
                -v "$(pwd)/models:/app/models" \
                -v "$(pwd)/uploads:/app/uploads" \
                $DOCKER_IMAGE web
            ;;

        2)
            ensure_dirs
            echo "Starting CLI version..."
            
            # Only pull if image doesn't exist locally
            if ! docker image inspect $DOCKER_IMAGE >/dev/null 2>&1; then
                echo "Image not found locally, pulling from Docker Hub..."
                docker pull $DOCKER_IMAGE
            fi

            docker run -it --rm \
                $GPU_FLAG \
                -v "$(pwd)/models:/app/models" \
                -v "$(pwd)/uploads:/app/uploads" \
                $DOCKER_IMAGE cli
            ;;

        0)
            echo "Goodbye!"
            exit 0
            ;;

        *)
            echo "Invalid option"
            ;;
    esac

    echo
    read -rp "Press Enter to continue..."
done