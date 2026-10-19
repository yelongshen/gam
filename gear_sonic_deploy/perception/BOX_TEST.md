# Box test: the onboard map against an exact map

Measures how the onboard terrain map differs from the map the sim would capture of the same
scene, on a geometry simpler than stairs: one box of known size on flat floor. The numbers
(height noise and bias, edge blur, blind zone, apparent drift, map and pose age, and the error
in the policy inputs) are what the sim's map corruption should reproduce.

## Setup

- A rigid box or plywood step, 0.1-0.3 m high, at least 0.3 x 0.3 m, with a flat top. Measure
  length, width and height to the millimetre.
- Flat floor, nothing else within 1.5 m of the box or the robot (people step out of the window).
- Robot standing (or on the gantry with `GANTRY_FILTER=true`), facing the box 0.6-1.0 m away.

## Recording (on the robot)

```bash
cd ~/gam-height-map/gear_sonic_deploy
./perception/jetson/run_onboard.sh start
# wait ~15 s: the map is stale for the first ~10 s while the mapper waits for odometry
python3 perception/record_terrain.py --seconds 60 --out ~/box_<name>.npz
./perception/jetson/run_onboard.sh stop
```

One recording per condition, e.g.:

1. `static`: robot standing still, 60 s.
2. `approach`: robot walked or stepped slowly toward the box and back (teleop or the gantry
   rail), 60 s. This exercises the blind zone and the map's memory of ground it has passed.
3. `with_policy`: as `static`, with the policy loaded on the Jetson (map age under GPU load).

## Analysis (workstation, humanoid-foundation-model branch `daphne/box-map-check`)

```bash
scp unitree@192.168.8.227:box_static.npz .
python gear_sonic/scripts/compare_box_map.py box_static.npz --box <length> <width> <height> --json box_static.json
```

It fits the floor and the box's pose (the box is placed by eye, so its pose is not needed),
then reports, against the exact map of that box:

- floor and box-top height errors (bias, noise, p95, outliers) away from the edges;
- the edge profile and its 10-90% rise width;
- the fraction of cells seen by distance from the torso, on the box top and in its shadow, and
  the radius from which half the cells are seen (the blind zone);
- apparent box motion between 5 s windows (odometry drift plus viewpoint effects: about 3 cm
  even without drift when the robot walks up to the box);
- map and pose age;
- the error of the policy inputs computed from these messages (pelvis scan `height_map_flat`,
  pelvis grid `pelvis_heightmap_11`) against the same messages carrying the exact map;
- the measurements next to the sim's `MapCorruption` default ranges.

Tested on synthetic recordings with known noise, bias, edge spill, blind zone, shadow and drift
(`gear_sonic/tests/test_compare_box_map.py`), and end to end through this recorder.
