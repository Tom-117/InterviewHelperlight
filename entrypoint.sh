#!/bin/bash
set -e


download_models() {
    if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models)" ]; then
        echo "First run detected. Downloading models..."
        python -c "from model_loader import load_local_models; load_local_models()"
    fi
}

case "$1" in
    web)
        shift
        download_models
        echo "Starting Flask application..."
        exec python app.py "$@"
        ;;

    cli)
        shift
        download_models
        echo "Running CLI version..."
        exec python main.py "$@"
        ;;

    shell)
        shift
        echo "Starting interactive shell..."
        exec /bin/bash
        ;;

    *)
        echo "Usage: {web|cli|shell} [args...]"
        echo "  web   - start the web UI (development mode)"
        echo "  cli   - run the command-line tool"
        echo "  shell - start an interactive shell"
        exit 1
        ;;
esac