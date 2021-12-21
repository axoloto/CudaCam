set -eo pipefail

printf "======================= ENV DEFINITION ========================== \n"

DEV_DIR=$(pwd)
DEV_BUILD_DIR="$DEV_DIR/build"
INSTALL_DIR="install"
DEV_INSTALL_DIR="$DEV_DIR/$INSTALL_DIR"

paths=" - root folder = $DEV_DIR\n"
paths+=" - build folder = $DEV_BUILD_DIR\n"
paths+=" - install folder = $DEV_INSTALL_DIR\n"
printf "%b" "$paths\n"

# Choosing Build Mode Release/Debug/RelWithDebInfo
if [[ -z "$1" ]] 
then
    echo "No build specified, using Release by default"
    DEV_BUILD_TYPE="Release"
elif [[ ( "$1" != "Release" && "$1" != "Debug" && "$1" != "RelWithDebInfo") ]]
then
    echo "$1 is not a supported build type, using Release instead"
    DEV_BUILD_TYPE="Release"
else
    DEV_BUILD_TYPE="$1"
fi

printf "========================= CMAKE SETUP ============================ \n"

mkdir -p "$DEV_INSTALL_DIR/$DEV_BUILD_TYPE"

# Preparing Cmake folder
cmake -S "$DEV_DIR" -B "$DEV_BUILD_DIR" \
 -DCMAKE_BUILD_TYPE="$DEV_BUILD_TYPE" \
 -DCMAKE_INSTALL_PREFIX="$DEV_INSTALL_DIR/$DEV_BUILD_TYPE"

printf "========================= CMAKE BUILD ============================ \n"

# Building app
cmake --build "$DEV_BUILD_DIR" --config "$DEV_BUILD_TYPE" 

printf "======================== CMAKE INSTALL =========================== \n"

# Installing app
cmake --install "$DEV_BUILD_DIR" --config "$DEV_BUILD_TYPE" 

printf "========================== START APP ============================= \n"

# Launching app
if [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "msys" ]]; then
    ./"$INSTALL_DIR/$DEV_BUILD_TYPE"/bin/CudaCam.exe
else
    ./"$INSTALL_DIR/$DEV_BUILD_TYPE"/bin/CudaCam
fi