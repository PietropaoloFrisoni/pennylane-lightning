cmake_minimum_required(VERSION 3.20)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░████████████████████████
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░████████████████████████
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░████████████████████████
]=])
message(${LOGO})

project(lightning_tensor
    DESCRIPTION "PennyLane Lightning Tensor C++ Backend."
    LANGUAGES CXX C
)

##########################
## Enforce C++ Standard ##
##########################
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

#########################
## Set Default Options ##
#########################
option(PLKOKKOS_ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(PLKOKKOS_ENABLE_SANITIZER "Enable address sanitizer" OFF)
option(PLKOKKOS_ENABLE_WARNINGS "Enable warnings" ON)
