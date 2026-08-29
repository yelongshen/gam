import argparse
import time
import zmq
import numpy as np
import glob
import os

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_dir", type=str, required=True, help="Directory with recorded NPZ frames")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port to publish on")
    parser.add_argument("--fps", type=float, default=50.0, help="Replay speed")
    parser.add_argument("--topic", type=str, default="pose", help="ZMQ topic")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.replay_dir, "*.npz")))
    if not npz_files:
        print(f"No npz files found in {args.replay_dir}")
        return
        
    print(f"Loaded {len(npz_files)} frames from {args.replay_dir}")
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # Important logic matching pico_manager precisely:
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.SNDBUF, 4096 * 1024)
    # Try different binds for safe network transmission
    try:
        socket.bind(f"tcp://127.0.0.1:{args.port}")
    except:
        socket.bind(f"tcp://*:{args.port}")
    
    print(f"ZMQ publisher bound on port {args.port}, topic '{args.topic}'")
    loop_time = 1.0 / args.fps
    
    print("Pre-loading files into memory to avoid I/O blocking during replay...")
    cache = []
    
    for i, f in enumerate(npz_files):
        dat = np.load(f)
        numpy_data = {}
        for k in dat.files:
            numpy_data[k] = dat[k]
            
        # The C++ policy specifically ingests the pack_pose_message structure (a strict 1024 byte padded JSON header string followed by binary chunks)
        packed_message = pack_pose_message(numpy_data, topic=args.topic)
        cache.append(packed_message)
        if i % 1000 == 0 and i > 0: print(f"Cached {i} frames...")
        
    print(f"Cached {len(cache)} frames.")
    
    print("---------------------------------------------------------")
    print("READY! The custom C++ binary format header is accurately replicated.")
    print("We are bypassing pico_manager and feeding directly to g1_deploy_onnx_ref.")
    print("---------------------------------------------------------")
    input("Press ENTER to start...")
    print("Starting... Go switch to the C++ binary!")
    
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
                
            if idx > 0 and idx % 1000 == 0:
                print(f"Replayed {idx} frames...")
                
    except KeyboardInterrupt:
        print("Replay stopped.")

if __name__ == "__main__":
    main()
