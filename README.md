# RINS-robot-

## Face_detection.py
Face_detection.py publisha koordinate obrazov na map gridu na topic **"/face_coords"**. Tip sporocila je definiran v **msg/FaceCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids**.

## detect_rings.py
detect_rings.py publisha koordinate ringov na map gridu na topic **"/ring_coords"**. Tip sporocila je definiran v **msg/RingCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids** ter **string[] colors**.

## Speech_servicer.py
Nudi 2 servica:
- **"/greet_service"** in **"/sayColor_service"**
- uporabljata tip sporocila **Speech.srv** je iz **string data ||| bool success**, v data napies kar hoces da rece
