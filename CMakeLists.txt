cmake_minimum_required(VERSION 2.8.3)
project(adaptive_clustering)

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs sensor_msgs visualization_msgs geometry_msgs pcl_conversions pcl_ros jsk_recognition_msgs)

find_package(PCL REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS})

add_message_files(FILES ClusterArray.msg)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package()

catkin_install_python(PROGRAMS
    scripts/pcl_pub.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(adaptive_clustering src/adaptive_clustering.cpp)
target_link_libraries(adaptive_clustering ${catkin_LIBRARIES})
if(catkin_EXPORTED_TARGETS)
  add_dependencies(adaptive_clustering ${catkin_EXPORTED_TARGETS})
endif()
