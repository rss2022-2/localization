cmake_minimum_required(VERSION 3.1)

project(localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  sensor_msgs
  std_msgs
  tf
  message_generation
)

add_message_files(
   FILES
   PoseError.msg
)



catkin_python_setup()

generate_messages(
   DEPENDENCIES
   std_msgs
#   ackermann_msgs#   geometry_msgs#   sensor_msgs#   std_msgs#   visualization_msgs
)