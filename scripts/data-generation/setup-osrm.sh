#!/bin/bash
# Setup OSRM with NYC OpenStreetMap data for local routing
#
# This script downloads NYC OSM data and sets up OSRM in Docker.
# OSRM will run on port 5000 and handle routing requests locally.
#
# Usage:
#   ./setup-osrm.sh          # Full setup (download + process + run)
#   ./setup-osrm.sh download # Just download OSM data
#   ./setup-osrm.sh process  # Just process (requires downloaded data)
#   ./setup-osrm.sh run      # Just run server (requires processed data)
#   ./setup-osrm.sh stop     # Stop the OSRM container

set -e

# Configuration
OSRM_VERSION="v5.27.1"
DATA_DIR="$(pwd)/osrm-data"
OSM_FILE="new-york-latest.osm.pbf"
CONTAINER_NAME="osrm-nyc"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        echo "  https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    log_info "Docker is available"
}

download_osm() {
    log_info "Downloading NYC OpenStreetMap data..."
    mkdir -p "$DATA_DIR"

    if [ -f "$DATA_DIR/$OSM_FILE" ]; then
        log_warn "OSM file already exists: $DATA_DIR/$OSM_FILE"
        read -p "Re-download? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    # Download from Geofabrik (New York state extract)
    # This includes all of NYC and surrounding areas
    curl -L -o "$DATA_DIR/$OSM_FILE" \
        "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"

    log_info "Downloaded OSM data to $DATA_DIR/$OSM_FILE"
}

process_osm() {
    log_info "Processing OSM data with OSRM..."

    if [ ! -f "$DATA_DIR/$OSM_FILE" ]; then
        log_error "OSM file not found: $DATA_DIR/$OSM_FILE"
        log_error "Run: $0 download"
        exit 1
    fi

    # Extract
    log_info "Step 1/3: Extracting road network..."
    docker run --rm -t \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-extract -p /opt/car.lua /data/$OSM_FILE

    # Partition
    log_info "Step 2/3: Partitioning for MLD algorithm..."
    docker run --rm -t \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-partition /data/${OSM_FILE%.osm.pbf}.osrm

    # Customize
    log_info "Step 3/3: Customizing routing weights..."
    docker run --rm -t \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-customize /data/${OSM_FILE%.osm.pbf}.osrm

    log_info "OSRM data processing complete!"
}

run_osrm() {
    log_info "Starting OSRM server..."

    # Check if already running
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        log_warn "OSRM container is already running"
        echo "  Stop it with: $0 stop"
        return
    fi

    # Check if data is processed
    if [ ! -f "$DATA_DIR/${OSM_FILE%.osm.pbf}.osrm.mldgr" ]; then
        log_error "OSRM data not processed. Run: $0 process"
        exit 1
    fi

    # Remove stopped container if exists
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    # Start server
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p 5000:5000 \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-routed --algorithm mld /data/${OSM_FILE%.osm.pbf}.osrm

    log_info "OSRM server started on http://localhost:5000"
    echo ""
    echo "Test it with:"
    echo "  curl 'http://localhost:5000/route/v1/driving/-73.99,40.73;-73.98,40.74?overview=false'"
    echo ""
    echo "Stop it with:"
    echo "  $0 stop"
}

stop_osrm() {
    log_info "Stopping OSRM server..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    log_info "OSRM server stopped"
}

full_setup() {
    check_docker
    download_osm
    process_osm
    run_osrm
}

# Show usage
usage() {
    echo "OSRM Setup Script for NYC Rideshare Data Generation"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  (none)    Full setup: download, process, and run"
    echo "  download  Download NYC OpenStreetMap data"
    echo "  process   Process OSM data for OSRM (takes ~10-15 minutes)"
    echo "  run       Start the OSRM routing server"
    echo "  stop      Stop the OSRM server"
    echo "  status    Check if OSRM is running"
    echo ""
    echo "Requirements:"
    echo "  - Docker"
    echo "  - ~5GB disk space for OSM data and processed files"
    echo "  - ~4GB RAM for processing"
    echo ""
    echo "After setup, the OSRM server will be available at:"
    echo "  http://localhost:5000"
}

status_osrm() {
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        log_info "OSRM is running"
        echo ""
        echo "Container: $CONTAINER_NAME"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Status}}\t{{.Ports}}"
        echo ""
        echo "Test endpoint:"
        curl -s 'http://localhost:5000/route/v1/driving/-73.99,40.73;-73.98,40.74?overview=false' | head -c 200
        echo "..."
    else
        log_warn "OSRM is not running"
        echo "Start it with: $0 run"
    fi
}

# Main
case "${1:-}" in
    download)
        check_docker
        download_osm
        ;;
    process)
        check_docker
        process_osm
        ;;
    run)
        check_docker
        run_osrm
        ;;
    stop)
        stop_osrm
        ;;
    status)
        status_osrm
        ;;
    help|--help|-h)
        usage
        ;;
    "")
        full_setup
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac

