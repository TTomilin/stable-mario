#!/usr/bin/env bash
set -e

NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'

REPO_ROOT=$( dirname ${BASH_SOURCE[0]} )/../..
DOCKERFILES_DIR=$( dirname ${BASH_SOURCE[0]} )
GENERATED_DOCKERFILES_DIR=$( dirname ${BASH_SOURCE[0]} )/tmp_dockerfiles
IMAGE_PREFIX="stable-retro_wheels"


# Array in format "<base docker image> <base dockerfile to use> <additional commands to add to the dockerfile after FROM statement>"
DOCKERFILES_TO_BUILD_AND_RUN=(
    "debian:11 apt-based.Dockerfile ENV LANG C.UTF-8"  # Python 3.9
    "debian:latest apt-based.Dockerfile ENV LANG C.UTF-8"  # Python 3.11
    "ubuntu:20.04 apt-based.Dockerfile"  # Python 3.8
    "ubuntu:22.04 apt-based.Dockerfile"  # Python 3.10
    "ubuntu:23.04 apt-based.Dockerfile"  # Python 3.11
    "ubuntu:latest apt-based.Dockerfile"  # Python 3.11
    "continuumio/miniconda3:latest conda-based.Dockerfile"  # Python 3.11
    #"almalinux:9 dnf-based.Dockerfile"  # Python 3.9  - test doesn't work becouse of pyglet requirement for X server
    #"rockylinux:9 dnf-based.Dockerfile"  # Python 3.9 - as above
    #"fedora:36 dnf-based.Dockerfile"  # Python 3.10 - as above
    #"fedora:37 dnf-based.Dockerfile"  # Python 3.11 - as above
)

# Clean local directory to avoid problems
cd $REPO_ROOT
rm -f CMakeCache.txt
rm -rf CMakeFiles
rm -f retro/*.so retro/cores/*.so retro/cores/*.json retro/cores/*-version
rm -f cores/*/*.so cores/snes/libretro/*.so
rm -rf build

# Build wheels using cibuildwheel
#export CIBW_BUILD_VERBOSITY=3  # Uncomment to see full build logs
cibuildwheel --platform linux --arch $(uname -m)

function create_dockerfile ( ) {
    local all_args=("$@")
    local base_image=$1
    local base_name=$( basename "$( echo ${base_image} | tr ':' '_' )" )
    local base_dockerfile=$2
    local add_commands=("${all_args[@]:2}")

    mkdir -p $GENERATED_DOCKERFILES_DIR
    dockerfile=${GENERATED_DOCKERFILES_DIR}/${IMAGE_PREFIX}_${base_name}.Dockerfile

    echo "FROM $base_image" > $dockerfile
    echo "" >> $dockerfile
    echo -e "${add_commands[@]}" >> $dockerfile
    cat ${DOCKERFILES_DIR}/$base_dockerfile | tail -n +2 >> $dockerfile
}

for dockerfile_setting in "${DOCKERFILES_TO_BUILD_AND_RUN[@]}"; do
    create_dockerfile $dockerfile_setting

    echo -n "Building and running $dockerfile, saving output to $dockerfile.log ... "
    filename=$( basename "$dockerfile" )
    dockerfile_dir=$( dirname "$dockerfile" )
    without_ext="${filename%.*}"
    tag="${without_ext}:latest"
    log="${dockerfile_dir}/${without_ext}.log"

    docker build -t $tag -f $dockerfile . &> $log || ( echo -e "${RED}FAILED${NC}"; exit 1 )
    docker run -it $tag &>> $log || ( echo -e "${RED}FAILED${NC}"; exit 1 )

    echo -e "${GREEN}OK${NC}"
done
