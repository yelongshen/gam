# Walking test: onboard perception while the G1 walks

Checks that the onboard perception stack (DLIO odometry, fused height map, `terrain` topic; see
[README.md](README.md), "Onboard") stays accurate while the robot walks, and that it doesn't disturb
gear_sonic's walking controller, which now shares the Jetson with it. No policy reads the terrain
topic yet, so the controller walks exactly as it does without perception. This test is about the
perception side and about CPU contention.

**Roles:** whoever normally runs the walking controller starts and stops it as usual; nothing here
changes how. Everything else below runs on the robot, from `~/gam` (branch `daphne/debug-depth`).

## 0. Before you start
1. **Baseline (recommended).** Do one short walk with perception stopped
   (`gear_sonic_deploy/perception/jetson/run_onboard.sh stop`) and note any warnings in the controller's
   console. g1_deploy_onnx_ref warns when its low-level state arrives late (>50 ms) or goes missing.
   That walk is the reference for step 5.
2. **Mark the start pose** on the floor with tape, position and heading. It's the ground truth for
   the return check.
3. **Clear the area.** Keep people more than about 1 m from the robot's path, since anyone nearby
   is mapped as an obstacle. Have your usual spotter and emergency stop ready.
4. The robot is **off the gantry**: do not use `GANTRY_FILTER` (it would hide real obstacles next
   to the robot).

## 1. Start perception (robot standing still)
```bash
cd ~/gam
LOG_DIR=~/g1_perception_logs gear_sonic_deploy/perception/jetson/run_onboard.sh start
```
Keep the robot still for about 10 s while DLIO calibrates its IMU. Then check it's healthy:
```bash
LOG_DIR=~/g1_perception_logs gear_sonic_deploy/perception/jetson/run_onboard.sh status
#   bridge: lidar ~10 Hz, depth ~15 Hz, imu ~175 Hz;  dlio: Distance to Origin < 0.01 m
python3 gear_sonic_deploy/scripts/terrain_subscriber.py
#   ~50 Hz; the cross-check line should read a few mm
```

## 2. Start recording
```bash
cd ~/gam; L=~/g1_perception_logs; T=$(date +%Y%m%d_%H%M%S); echo "recording walk_$T"
# LiDAR, IMU, DLIO pose, TF (inside the container; ~2 MB/s)
docker exec -d g1_perception bash -lc "source /opt/ros/humble/install/setup.bash && source /ws/install/setup.bash && timeout -s INT 900 ros2 bag record -o /logs/walk_$T /g1/lidar/points /g1/imu /dlio/odom /tf /tf_static > /logs/walk_${T}_bag.log 2>&1"
# the terrain topic, every message
setsid nohup timeout 900 python3 gear_sonic_deploy/scripts/terrain_subscriber.py --seconds 900 --record $L/walk_${T}_terrain.npz > $L/walk_${T}_terrain.log 2>&1 < /dev/null &
# CPU / GPU / RAM / temperature, once per second
setsid nohup timeout 900 tegrastats --interval 1000 > $L/walk_${T}_tegrastats.log 2>&1 < /dev/null &
```
Everything stops by itself after 15 min (`timeout 900`). Recording adds little load: measured on
the robot, CPU 0 at about 33% median, cores 1-3 about 19%, cores 4-7 (perception) about 46%.

## 3. Walk
1. Start the walking controller as usual and stand for about 10 s.
2. Walk **slowly forward about 2 m**, then stop for 10 s.
3. **Turn about 90°**, then stop for 10 s.
4. **Walk back to the tape mark**, stop facing the original heading, and hold for 10 s. The still
   periods at the start and end are what the accuracy check compares.
5. If that went well, repeat once at normal walking speed (hold still again at the end).
6. Stop the controller as usual.

If DLIO diverges mid-walk (`run_onboard.sh status` shows the position jumping to metres), stop
walking. It doesn't affect the controller, which doesn't read perception. Keep the recording for
analysis.

## 4. Stop recording
```bash
docker exec g1_perception pkill -INT -f "[r]os2 bag record"
pkill -INT -f "[t]errain_subscriber.py"      # writes walk_<T>_terrain.npz on exit
pkill -f "[t]egrastats --interval"
LOG_DIR=~/g1_perception_logs gear_sonic_deploy/perception/jetson/run_onboard.sh stop   # optional
```
(The `[x]` in each pattern stops `pkill` from matching its own command line.)

## 5. Analyze
```bash
# terrain topic (rate, staleness, window coverage over time, distance walked) + CPU/GPU/RAM
python3 gear_sonic_deploy/perception/analyze_walk.py ~/g1_perception_logs/walk_$T
# DLIO accuracy: each still period vs the first, by ICP on the raw scans (independent of DLIO)
docker exec g1_perception bash -lc "source /opt/ros/humble/install/setup.bash && source /ws/install/setup.bash && python3 /perception/eval_motion.py /logs/walk_$T /logs/walk_${T}_eval"
```
To look at the data elsewhere, copy `~/g1_perception_logs/walk_<T>*` off the robot.
The bag directory is written by the container as root: delete old ones with
`docker exec g1_perception rm -rf /logs/walk_<T>` (or `sudo rm -rf`).
`eval_motion.py` also runs in the workstation container, which reads the robot's sqlite3 bags.

| Question | Where to look | Pass |
|---|---|---|
| Does odometry survive walking? | `eval_motion.py`: DLIO error at each still period; `analyze_walk.py`: start/end distance vs the tape mark | within a few cm and about 1° (gantry: 4-10 mm, <0.35°) |
| Is the policy input usable while moving? | `analyze_walk.py`: rate, gaps, staleness, window coverage | ~50 Hz, no gaps > 100 ms; window fills in under and behind the feet after walking forward (static: ~36%) |
| Is the map clean? | window median per 10 s in `analyze_walk.py` (should stay near the torso height); RViz or `dump_gridmap.py` for a picture | floor within about 2 cm, no smearing or doubled edges |
| Is the controller unaffected? | the controller's console vs the baseline walk; `analyze_walk.py` CPU per core (c0 = control thread) | no new late or missing-state warnings; c0 well below saturation |

Reference numbers from the gantry tests are in README.md ("Validation" and "Onboard").
