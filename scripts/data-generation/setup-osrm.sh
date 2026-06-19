#!/bin/bash
# Setup OSRM with OpenStreetMap data for local routing.
#
# Downloads an OSM extract and processes it with OSRM in Docker, then runs the
# routing server. Region and routing profile are selected by environment vars so
# multiple servers can coexist (e.g. NYC car on :5000, Montréal bicycle on :5001).
#
# Config (env vars, all optional):
#   REGION          new-york (default) | quebec
#   PROFILE         routing profile: car | bicycle | foot (default per region)
#   PORT            host port for osrm-routed       (default per region)
#   CONTAINER_NAME  docker container name           (default per region)
#
# Usage:
#   ./setup-osrm.sh                                  # NYC car (download+process+run)
#   REGION=quebec PROFILE=bicycle ./setup-osrm.sh    # Montréal BIXI bike network
#   REGION=quebec PROFILE=bicycle ./setup-osrm.sh download   # just download
#   REGION=quebec PROFILE=bicycle ./setup-osrm.sh process    # just process
#   REGION=quebec PROFILE=bicycle ./setup-osrm.sh run        # just run server
#   REGION=quebec ./setup-osrm.sh stop                       # stop the container
#   REGION=quebec ./setup-osrm.sh status                     # check status

set -e

# Configuration
OSRM_VERSION="v5.27.1"
DATA_DIR="$(pwd)/osrm-data"

# --- Region presets --------------------------------------------------------
REGION="${REGION:-new-york}"
case "$REGION" in
    new-york|nyc)
        REGION="new-york"
        OSM_FILE="new-york-latest.osm.pbf"
        OSM_URL="https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"
        DEFAULT_PROFILE="car"
        DEFAULT_PORT="5000"
        DEFAULT_CONTAINER="osrm-nyc"
        # A short Manhattan leg — used only to assert the server answers "Ok".
        TEST_ROUTE="-73.99,40.73;-73.98,40.74"
        ;;
    quebec|montreal|mtl)
        REGION="quebec"
        OSM_FILE="quebec-latest.osm.pbf"
        OSM_URL="https://download.geofabrik.de/north-america/canada/quebec-latest.osm.pbf"
        DEFAULT_PROFILE="bicycle"
        DEFAULT_PORT="5001"
        DEFAULT_CONTAINER="osrm-montreal-bike"
        # A short downtown-Montréal leg.
        TEST_ROUTE="-73.59,45.50;-73.58,45.51"
        ;;
    *)
        echo "Unknown REGION '$REGION' (expected: new-york | quebec)" >&2
        exit 1
        ;;
esac

PROFILE="${PROFILE:-$DEFAULT_PROFILE}"
PORT="${PORT:-$DEFAULT_PORT}"
CONTAINER_NAME="${CONTAINER_NAME:-$DEFAULT_CONTAINER}"

case "$PROFILE" in
    car)     PROFILE_LUA="/opt/car.lua" ;;
    bicycle) PROFILE_LUA="/opt/bicycle.lua" ;;
    foot)    PROFILE_LUA="/opt/foot.lua" ;;
    *)
        echo "Unknown PROFILE '$PROFILE' (expected: car | bicycle | foot)" >&2
        exit 1
        ;;
esac

OSRM_BASE="${OSM_FILE%.osm.pbf}"  # e.g. quebec-latest

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

log_config() {
    log_info "Region: $REGION | Profile: $PROFILE ($PROFILE_LUA) | Port: $PORT | Container: $CONTAINER_NAME"
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
    log_info "Downloading $REGION OpenStreetMap data..."
    mkdir -p "$DATA_DIR"

    if [ -f "$DATA_DIR/$OSM_FILE" ]; then
        log_warn "OSM file already exists: $DATA_DIR/$OSM_FILE"
        read -p "Re-download? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    curl -L -o "$DATA_DIR/$OSM_FILE" "$OSM_URL"

    log_info "Downloaded OSM data to $DATA_DIR/$OSM_FILE"
}

process_osm() {
    log_info "Processing OSM data with OSRM ($PROFILE profile)..."

    if [ ! -f "$DATA_DIR/$OSM_FILE" ]; then
        log_error "OSM file not found: $DATA_DIR/$OSM_FILE"
        log_error "Run: $0 download"
        exit 1
    fi

    # Extract (the PROFILE is baked in here, at extract time).
    log_info "Step 1/3: Extracting road network with $PROFILE_LUA..."
    docker run --rm -t \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-extract -p "$PROFILE_LUA" "/data/$OSM_FILE"

    # Partition
    log_info "Step 2/3: Partitioning for MLD algorithm..."
    docker run --rm -t \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-partition "/data/${OSRM_BASE}.osrm"

    # Customize
    log_info "Step 3/3: Customizing routing weights..."
    docker run --rm -t \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-customize "/data/${OSRM_BASE}.osrm"

    log_info "OSRM data processing complete!"
}

run_osrm() {
    log_info "Starting OSRM server..."

    # Check if already running
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        log_warn "OSRM container '$CONTAINER_NAME' is already running"
        echo "  Stop it with: $0 stop"
        return
    fi

    # Check if data is processed
    if [ ! -f "$DATA_DIR/${OSRM_BASE}.osrm.mldgr" ]; then
        log_error "OSRM data not processed. Run: $0 process"
        exit 1
    fi

    # Remove stopped container if exists
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    # Start server
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${PORT}:5000" \
        -v "$DATA_DIR:/data" \
        ghcr.io/project-osrm/osrm-backend:${OSRM_VERSION} \
        osrm-routed --algorithm mld "/data/${OSRM_BASE}.osrm"

    log_info "OSRM server started on http://localhost:${PORT}"
    echo ""
    echo "Test it with:"
    echo "  curl 'http://localhost:${PORT}/route/v1/driving/${TEST_ROUTE}?overview=false'"
    echo ""
    echo "Stop it with:"
    echo "  $0 stop"
}

stop_osrm() {
    log_info "Stopping OSRM server '$CONTAINER_NAME'..."
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
    echo "OSRM Setup Script (multi-region, multi-profile)"
    echo ""
    echo "Usage: [REGION=..] [PROFILE=..] [PORT=..] $0 [command]"
    echo ""
    echo "Config (env vars):"
    echo "  REGION    new-york (default) | quebec"
    echo "  PROFILE   car | bicycle | foot   (default: car for new-york, bicycle for quebec)"
    echo "  PORT      host port               (default: 5000 new-york, 5001 quebec)"
    echo ""
    echo "Commands:"
    echo "  (none)    Full setup: download, process, and run"
    echo "  download  Download the region's OpenStreetMap data"
    echo "  process   Process OSM data for OSRM (takes ~10-15 minutes)"
    echo "  run       Start the OSRM routing server"
    echo "  stop      Stop the OSRM server"
    echo "  status    Check if OSRM is running"
    echo ""
    echo "Examples:"
    echo "  $0                                          # NYC car on :5000"
    echo "  REGION=quebec PROFILE=bicycle $0            # Montréal BIXI bike on :5001"
    echo ""
    echo "Requirements:"
    echo "  - Docker"
    echo "  - ~5GB disk space for OSM data and processed files"
    echo "  - ~4GB RAM for processing"
}

status_osrm() {
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        log_info "OSRM '$CONTAINER_NAME' is running"
        echo ""
        echo "Container: $CONTAINER_NAME"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Status}}\t{{.Ports}}"
        echo ""
        echo "Test endpoint:"
        curl -s "http://localhost:${PORT}/route/v1/driving/${TEST_ROUTE}?overview=false" | head -c 200
        echo "..."
    else
        log_warn "OSRM '$CONTAINER_NAME' is not running"
        echo "Start it with: $0 run"
    fi
}

# Main
log_config
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
