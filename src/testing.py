import zmq
import cv2
import json
import numpy as np

# 1. Initialize ZMQ Subscriber / Pull Socket
context = zmq.Context()
socket = context.socket(zmq.PULL) # Match the sender architecture (PUSH -> PULL or PUB -> SUB)

# Port configured on Quest 3 UI
PORT = 8099
socket.bind(f"tcp://*:{PORT}")
print(f"[Receiver] Listening for Quest 3 stream on port {PORT}...")

frame_count = 0
latest_frame_id = 0

try:
    while True:
        try:
            # Receive multipart message: [Metadata JSON, Raw JPG Bytes]
            metadata_bytes, left_image_bytes, right_image_bytes = socket.recv_multipart()
            # Parse Metadata (Intrinsics + Pose)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            frame_id = metadata['frame_id']
            
            if latest_frame_id > int(frame_id):
                print(f"Skipping this frame as this is frame {frame_id} but we already saw frame {latest_frame_id}")
                continue
            
            frame_timestamp = metadata['timestamp_us']
            left, right = metadata['leftCamera'], metadata['rightCamera']
            
            left_fx, left_fy = left['fx'], left['fy']
            right_fx, right_fy = right['fx'], right['fy']
            left_cx, left_cy = left['cx'], left['cy']
            right_cx, right_cy = right['cx'], right['cy']
            left_height, left_width = left['shape']
            right_height, right_width = right['shape']
            left_pos, left_rot = left['pos'], left['rot']
            right_pos, right_rot = right['pos'], right['rot']
            
            
            # Decode JPEG Frame
            np_left_arr = np.frombuffer(left_image_bytes, np.uint8)
            left_frame_bgr = cv2.imdecode(np_left_arr, cv2.IMREAD_COLOR)

            np_right_arr = np.frombuffer(right_image_bytes, np.uint8)
            right_frame_bgr = cv2.imdecode(np_right_arr, cv2.IMREAD_COLOR)
        
            # Safety check: skip if frame decoding failed
            if left_frame_bgr is None or right_frame_bgr is None:
                print("[Warning] Received invalid or corrupted image frame bytes. Skipping...")
                continue
        
            # Ensure vertical heights match before horizontal stacking
            if left_frame_bgr.shape[0] != right_frame_bgr.shape[0]:
                right_frame_bgr = cv2.resize(
                    right_frame_bgr, 
                    (left_frame_bgr.shape[1], left_frame_bgr.shape[0])
                )
            
            skew_us = abs(left['timestamp_us'] - right['timestamp_us'])
            cv2.imwrite(f'sample_input/streamed_frames/left/left_{frame_id:06d}.jpg', left_frame_bgr)
            cv2.imwrite(f'sample_input/streamed_frames/left/right_{frame_id:06d}.jpg', right_frame_bgr)
            
            # Overlay telemetry on live laptop window
            cv2.putText(left_frame_bgr, f"Frame: {frame_count} | Position: ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f})", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(right_frame_bgr, f"Focal Length (fx, fy): ({right_fx:.1f}, {right_fy:.1f})", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(right_frame_bgr, f"Skew: {skew_us}",
                        (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Combine images side-by-side (Ensure shapes match)
            stereo_frame = np.hstack((left_frame_bgr, right_frame_bgr))

            # Show Live Window
            cv2.imshow("Quest 3 Stereo Stream", stereo_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            latest_frame_id = frame_id
        
        except Exception as e:
            print(f"[Error] Stream decoding error: {e}")

except KeyboardInterrupt:
    print(f"Stream stopped by KeyboardInterrupt")
        
finally:
    cv2.destroyAllWindows()
    socket.close()
    context.term()
    print("[Receiver] Socket cleanly closed.")