# TE3002B-Puzzlebot

## Puzzlebot Vision

Codigo para un Puzzlebot con camara frontal, seguimiento de carril por vision y reaccion a semaforos/senales mediante ROS 2 Humble.

El sistema esta separado en tres partes:

- `shared_camera_publisher.py`: abre la webcam una sola vez y publica `/camera/image_raw`.
- `direct_camera_lane_follower_recorder.cpp`: sigue la linea/carril, detecta zebra/interseccion y manda velocidades a las ruedas.
- `traffic_light_sign_publisher.py`: detecta semaforo con YOLO ONNX y publica `RED`, `YELLOW` o `GREEN` en `/sign_detected`.

## Estructura

```text
direct_camera_lane_follower_recorder.cpp
shared_camera_publisher.py
traffic_light_sign_publisher.py
test_track_lines.py
.gitignore
README.md
```

Los modelos (`.pt`, `.onnx`, `.engine`), videos y carpetas de resultados no se suben al repo.

## Build En Jetson

Copiar el nodo C++ al paquete ROS:

```bash
scp direct_camera_lane_follower_recorder.cpp \
  puzzlebot@10.22.2.225:/home/puzzlebot/ros2_ws/src/puzzlebot_cpp_vision/src/direct_camera_lane_follower_recorder.cpp
```

Copiar scripts Python:

```bash
scp shared_camera_publisher.py traffic_light_sign_publisher.py \
  puzzlebot@10.22.2.225:/home/puzzlebot/yolotest/
```

En la Jetson:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select puzzlebot_cpp_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
chmod +x ~/yolotest/shared_camera_publisher.py ~/yolotest/traffic_light_sign_publisher.py
```

Si el build falla por `sensor_msgs`, agregar `sensor_msgs` a `CMakeLists.txt` y `package.xml`.

## Ejecucion

Terminal 1: camara compartida.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

taskset -c 3 python3 ~/yolotest/shared_camera_publisher.py \
  --source 0 \
  --width 320 \
  --height 240 \
  --fps 30 \
  --publish-hz 15 \
  --topic /camera/image_raw
```

Terminal 2: seguidor de carril autonomo.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

taskset -c 0,1 ros2 run puzzlebot_cpp_vision direct_camera_lane_follower_recorder --ros-args \
  -p use_image_topic:=true \
  -p image_topic:=/camera/image_raw \
  -p speed_straight:=1.20 \
  -p speed_soft_turn:=0.95 \
  -p speed_hard_turn:=0.70 \
  -p search_speed:=0.45 \
  -p max_speed:=2.50 \
  -p seconds:=0.0 \
  -p side_by_side:=true
```

Modo monitor/sin motores. El nodo sigue viendo la camara, detectando
zebra/semaforo y grabando video, pero no publica comandos de movimiento:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

taskset -c 0,1 ros2 run puzzlebot_cpp_vision direct_camera_lane_follower_recorder --ros-args \
  -p use_image_topic:=true \
  -p image_topic:=/camera/image_raw \
  -p enable_wheel_commands:=false \
  -p disabled_startup_stop_frames:=12 \
  -p seconds:=0.0 \
  -p side_by_side:=true
```

`disabled_startup_stop_frames` manda `0,0` solo al arrancar para limpiar un
comando de motor anterior que haya quedado retenido. Despues de esos frames el
lane follower deja de publicar a las ruedas.

Terminal 3: detector de semaforo.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

taskset -c 2 python3 ~/yolotest/traffic_light_sign_publisher.py \
  --model /home/puzzlebot/yolotest/puzzlebot_best_v2.onnx \
  --image-topic /camera/image_raw \
  --hz 1 \
  --imgsz 320 \
  --conf 0.35 \
  --roi 0.0 0.0 0.75 0.85
```

## Topics Principales

```text
/camera/image_raw    sensor_msgs/Image
/sign_detected       std_msgs/String
/VelocitySetL        std_msgs/Float32
/VelocitySetR        std_msgs/Float32
```

## Debug

El video se guarda por default en:

```text
/home/puzzlebot/lane_drive_debug.avi
```

Con `side_by_side:=true`:

- izquierda: vista real de la camara sin overlays.
- derecha: mascara, ROI principal, zebra y estado compacto.

## Parametros Importantes

Velocidad:

```text
speed_straight
speed_soft_turn
speed_hard_turn
search_speed
max_speed
traffic_green_speed_multiplier
traffic_yellow_speed_multiplier
enable_wheel_commands
disabled_startup_stop_frames
```

ROI principal de carril:

```text
El poligono amarillo se ajusta en buildTrackMask() dentro de
direct_camera_lane_follower_recorder.cpp.
```

Coordenadas normalizadas:

```text
x = 0.0 izquierda, 1.0 derecha
y = 0.0 arriba,    1.0 abajo
```

## Git

Repositorio remoto:

```text
https://github.com/omarcavazosgzz/TE3002B-Puzzlebot.git
```
