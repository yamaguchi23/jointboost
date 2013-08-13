cmake_minimum_required (VERSION 2.6)

project (JointBoost)

set (CMAKE_BUILD_TYPE Release)

add_executable(jointtrain jointtrain.cpp readMultiClassDataFile.cpp JointBoost.cpp)
add_executable(jointpredict jointpredict.cpp readMultiClassDataFile.cpp JointBoost.cpp)
