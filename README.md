# RINS-robot-

## Face_detection.py
Face_detection.py publisha koordinate obrazov na map gridu na topic **"/face_coords"**. Tip sporocila je definiran v **msg/FaceCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids**.

## detect_rings.py
detect_rings.py publisha koordinate ringov na map gridu na topic **"/ring_coords"**. Tip sporocila je definiran v **msg/RingCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids** ter **string[] colors**.

## Speech_servicer.py
Nudi 2 servica:
- **"/greet_service"** in **"/sayColor_service"**
- uporabljata tip sporocila **Speech.srv** je iz **string data ||| bool success**, v data napies kar hoces da rece



run z:
1. ros2 run rmw_zenoh_cpp rmw_zenohd
2. ros2 launch rins_robot sim_turtlebot_nav.launch.py
3. ros2 run rins_robot arm_mover_actions.py
-spremembe z ros2 topic pub --once /arm_command std_msgs/msg/String "{data: look_at_belt_right}"
-izbira med:
    -look_at_belt_right
    -look_at_belt_left
    -look_for_qr
    -garage
    -up
    (-manual)
4. ros2 run rqt_image_view rqt_image_view /top_camera/rgb/preview/image_raw
-kamera na roki