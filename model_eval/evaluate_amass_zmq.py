import argparse
import time
import zmq
import numpy as np
import glob
import os
import msgpack
try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
except ImportError:
    print("Failed to import pack_pose_message. Ensure you are in the correct python env.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_dir", type=str, default="/home/grease/egodata/downloads/amass/extracted/ACCAD", help="Directory with raw AMASS NPZ frames")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port to publish on")
    parser.add_argument("--fps", type=float, default=50.0, help="Replay speed")
    parser.add_argument("--topic", type=str, default="pose", help="ZMQ topic")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.amass_dir, "**/*.npz"), recursive=True))
    if not npz_files:
        print(f"No npz files found in {args.amass_dir}")
        return
        
    print(f"Loaded {len(npz_files)} raw AMASS sequences.")
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.SNDBUF, 4096 * 1024)
    # Try different binds for safe network transmission
    try:
        socket.bind(f"tcp://127.0.0.1:{args.port}")
    except:
        socket.bind(f"tcp://*:{args.port}")
    
    print(f"ZMQ publisher bound on port {args.port}, topic '{args.topic}'. SIMULATING MODE 2...")
    loop_time = 1.0 / args.fps
    
    # We will pick ONE specific test sequence to stream continuously
    target_file = [f for f in npz_files if "CartWheel" in f][0]
    print(f"Selected Evaluation Clip: {os.path.basename(target_file)}")
    
    dat = np.load(target_file)
    poses = dat['poses']
    trans = dat['trans']
    num_frames = poses.shape[0]

    # In mode_id: 2, the Policy specifically expects 'smpl_joints', 'body_quat_w', 'joint_pos' inside the ZMQ payload
    # Even if they are blank, the keys must exist in the C++ struct parser.
    # The true SMPL geometric locations are calculated by the pico_manager normally 
    # but for offline streaming we can mathematically estimate or zero-pad the placeholders.
    print("Formatting payload into C++ expected message structure...")
    
    cache = []
    for i in range(num_frames):
        numpy_data = {
            "smpl_joints": np.zeros((4, 24, 3), dtype=np.float32),          # 4-frame window
            "body_quat_w": np.zeros((4, 4), dtype=np.float32),              # 4-frame window
            "joint_pos": np.zeros((4, 29), dtype=np.float32),               # 4-frame window (for hands/wrists)
            "joint_vel": np.zeros((1, 29), dtype=np.float32),
            "vr_position": trans[i].astype(np.float32),
            "vr_orientation": np.array([1, 0, 0, 0], dtype=np.float32),     # Dummy quat
            "frame_index": np.array([i], dtype=np.int32),
            "left_trigger": np.zeros(1, dtype=np.float32),
            "right_trigger": np.zeros(1, dtype=np.float32),
            "left_grip": np.zeros(1, dtype=np.float32),
            "right_grip": np.zeros(1, dtype=np.float32),
            "pico_dt": np.array([1.0/50.0], dtype=np.float64),
            "pico_fps": np.array([50.0], dtype=np.float64),
            "timestamp_realtime": np.array([time.time()], dtype=np.float64),
            "timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
            "left_hand_joints": np.zeros((7,), dtype=np.float32),
            "right_hand_joints": np.zeros((7,), dtype=np.float32),
            "toggle_data_collection": np.zeros(1, dtype=bool),
            "toggle_data_abort": np.zeros(1, dtype=bool),
            "heading_increment": np.zeros(1, dtype=np.float32)
        }
        
        # Inject the actual pure 72-dim pose data for the 4-frame lookahead chunk
        for f_offset in range(4):
            idx = min(i + f_offset, num_frames - 1)
            # Take the 72 AMASS pose dimensions, slice into 24x3 pseudo-joints representation
            amass_pose = poses[idx, :72].reshape(24, 3) 
            numpy_data["smpl_joints"][f_offset] = amass_pose
            
        packed_message = pack_pose_message(numpy_data, topic=args.topic)
        cache.append(packed_message)
        
    print(f"Cached {len(cache)} frames.")
    print("---------------------------------------------------------")
    print("READY TO EVALUATE: Open C++ Policy!")
    print("---------------------------------------------------------")
    input("Press ENTER to start...")
    
    idx = 0
    try:
        while True:
            t_start = time.time()
            socket.send(cache[idx])
            idx = (idx + 1) % len(cache)
            elapsed = time.time() - t_start
            delay = loop_time - elapsed
            if delay > 0:
                time.sleep(delay)
                
            if idx > 0 and idx % 100 == 0:
                print(f"Replayed {idx} frames of Cartwheel...")
    except KeyboardInterrupt:
        print("Replay stopped.")

if __name__ == "__main__":
    main()
