from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils import load_serialized_data

@dataclass 
class InputFrame:
    """
    Dataclass representing one frame of input, however it was obtained.
    """
    index: int                      # Pipeline index; contiguous even when the source drops
    left: np.ndarray                # shape (3, H, W) RGB frame at the target extent
    right: Optional[np.ndarray]     # (3, H, W) RGB frame at the target extent, or None
    metadata: Optional[Dict]        # Metadata capture per frame, or None
    frame_id: Optional[int] = None  # Frame id assigned by the source (headset frame number)
    dropped: int = 0                # Source frames skipped since the previous delivered frame

class FrameSource:
    """
    Base class for frame source.
    
    
    """
    
    def __init__(self, target_extent):
        self.target_extent = target_extent # (height, width)
        self.source_size = None             # (width, height) as received, before resizing
        self._pending: Optional[InputFrame] = None
        self._index = 0
    
    def __enter__(self):
            return self
        
    def __exit__(self, *exc):
        self.close()
        return False
        
    def __iter__(self):
        return self
        
    def __next__(self) -> InputFrame:
        """TODO: docstring"""
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        return self._next_frame()
    
    def bootstrap(self) -> InputFrame:
        """
        Return the first frame. Must be called before iterating.

        Returns:
            InputFrame: _description_
        """
        if self._pending is None:
            self._pending = self._next_frame()
        return self._pending
    
    def close(self):
        pass
    
    def _load_frame(self, frame, extent=None):
        """
        Load a frame as a torch tensor.

        Returns nparray representing frame (Height, Width, Channels).

        Args:
            frame: Path to a frame, or frame itself.
        """
        if isinstance(frame, str):
            frame = Image.open(frame).convert("RGB")
        
        frame_np = np.asarray(frame)
        if extent:
            frame_np = cv2.resize(frame_np, (extent[1], extent[0]), interpolation=cv2.INTER_LINEAR)

        frame_np_trans = np.transpose(frame_np, axes=(2, 0, 1))
        return frame_np_trans # shape: (3, H, W)
    
    def _next_frame(self) -> InputFrame:
        raise NotImplementedError
    
    def _take_index(self) -> int:
        index = self._index
        self._index += 1
        return index
    
    def _to_chw_rgb(self, image_bgr):
        """
        Decoded BGR (H, W, 3) -> (3, H, W) uint8 RGB at the target extent.

        Mirrors `utils.load_frame` so both sources hand the pipeline the same
        layout, dtype and interpolation.
        """
        height, width = self.target_extent
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(image.transpose(2, 0, 1))
    
class DirectoryFrameSource(FrameSource):
    """Offline frame source. Replay of a captured sequence."""
    
    def __init__(self, input, target_extent, input_right=None, input_metadata=None):
        super().__init__(target_extent)
        
        self._left_paths = self._sorted_dir(input)
        if not self._left_paths:
            raise RuntimeError(f"No frames found in {input}")
        
        self._right_paths = self._sorted_dir(input_right) if input_right is not None else None
        self._metadata_paths = self._sorted_dir(input_metadata) if input_metadata is not None else None
        
        if self._right_paths is not None:
            assert len(self._right_paths) == len(self._left_paths), \
                        f"Sequence length mismatch, left_paths has {len(self._left_paths)} frames but right_paths has {len(self._right_paths)} frames"
        if self._metadata_paths is not None:
            assert len(self._metadata_paths) == len(self._left_paths), \
                f"Sequence length mismatch, left_paths has {len(self._left_paths)} frames but metadata_paths has {len(self._metadata_paths)} frames"

        # Native resolution to compute the intrinsics scale factor
        _, source_height, source_width = self._load_frame(str(self._left_paths[0])).shape
        self.source_size = (source_width, source_height)
    
    def __len__(self):
        return len(self._left_paths)
    
    def _next_frame(self) -> InputFrame:
        i = self._index
        if i >= len(self._left_paths):
            raise StopIteration
        
        height, width = self.target_extent
        left = self._load_frame(str(self._left_paths[i]), extent=self.target_extent)
        right = self._load_frame(str(self._right_paths[i]), extent=self.target_extent) if self._right_paths is not None else None
        metadata = load_serialized_data(str(self._metadata_paths[i])) if self._metadata_paths is not None else None
        
        return InputFrame(index=self._take_index(), left=left, right=right, metadata=metadata, frame_id=i)
    
    @staticmethod
    def _sorted_dir(directory):
        
        def frames_sort_key(file):
            f = str(file)
            result = f.rsplit('_', 1)[-1]
            result = result.rsplit('.')[0]
            return int(result)
        
        return sorted([f for f in Path(directory).iterdir()], key=frames_sort_key)
    
class StreamedFrameSource(FrameSource):
    """Online frame source. Streamed from headset."""
    
    def __init__(self, target_extent, port=8099, connect=None, rcvhwm=10, first_frame_timeout=60.0, stall_warn=5.0):
        super().__init__(target_extent)
        # Imports here so offline path doesn't require it
        import zmq 
        import threading
        
        self._zmq = zmq
        self._port = port
        self._first_frame_timeout = first_frame_timeout
        self._stall_warn = stall_warn
        self._awaiting_first = True
        
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        # Receiver thead drains far faster than the headset sends.
        # Bound so the socket cannot hoard frames behind one-slot buffer
        self._socket.setsockopt(zmq.RCVHWM, rcvhwm)
        if connect is not None:
            self._socket.connect(connect)
        else:
            self._socket.bind(f"tcp://*:{port}")
            
        self._slot = None
        self._dropped = 0
        self.decode_failures = 0
        self._latest_frame_id = -1
        self._condition = threading.Condition()
        self._stop = threading.Event()
        
        # Create and start frame receiving thread
        self._thread = threading.Thread(target=self._receive_loop, name="frame-receiver", daemon=True)
        self._thread.start()
        print(f"[FrameSource] Listening for headset stream on port {port}...")
        
    def _receive_loop(self):
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        
        while not self._stop.is_set():
            try:
                # Polling rather than blocking keeps the thread responsive to close()
                if not poller.poll(200):
                    continue
                parts = self._socket.recv_multipart(zmq.NOBLOCK)
            
            # Keep receiving if we get this exception
            except zmq.Again:
                continue
            
            # Tear socket down during shutdown
            except (zmq.ContextTerminated, zmq.ZMQError):
                break
            
            # Try to decode the parts into metadata, left frame, right frame
            decoded = self._decode(parts)
            # Continue if decoding fails
            if decoded is None:
                continue
                
            with self._condition:
                # If there's something in the slot, pipeline is behind
                if self._slot is not None:
                    # Keep the newer frame and drop the older
                    self._dropped += 1
                self._slot = decoded
                self._condition.notify()
    
    def _decode(self, parts):
        # Check if correct number of parts
        if len(parts) != 3:
            self.decode_failures += 1
            return None
        
        # Decode
        metadata_bytes, left_bytes, right_bytes = parts
         
        try:
            metadata = load_serialized_data(metadata_bytes.decode("utf-8"), load_type='json')
            frame_id = int(metadata["frame_id"])
        except (ValueError, KeyError, UnicodeDecodeError) as error:
            print(f"[FrameSource] Malformed metadata, skipping frame: {error}")
            self.decode_failures += 1
            return None
        
        # If the arrival is out of order (a newer frame has already gone through), drop frame
        if frame_id <= self._latest_frame_id:
            return None
        
        left_np = np.frombuffer(left_bytes, np.uint8)
        right_np = np.frombuffer(right_bytes, np.uint8)
        left = self._load_frame(left_np)
        right = self._load_frame(right_np)
        
        # If the image bytes are corrupted, drop frame
        if left is None or right is None:
            print("[FrameSource] received invalid or corrupted image bytes, skipping frame")
            self.decode_failures += 1
            return None
        
        # Set source size if not already set
        if self.source_size is None:
            self.source_size = (left.shape[1], left.shape[0])
        
        # Set latest frame id to this frame id
        self._latest_frame_id = frame_id
        
        return frame_id, left, right, metadata
    
    def _next_frame(self) -> InputFrame:
        timeout = self._first_frame_timeout if self._awaiting_first else None
        waited = 0.0
        poll_s = 0.5
        
        with self._condition:
            # Keep iterating until we fill slot through _receive_loop()
            while self._slot is None:
                # If we're stopped, raise StopIteration
                if self._stop.is_set():
                    raise StopIteration
                
                # Short waits rather than one long one so Ctrl-C stays responsive
                self._condition.wait(timeout=poll_s)
                
                # Stop looking to fill slot if it has been filled
                if self._slot is not None:
                    break
                
                # Update amount of time waited
                waited += poll_s
                
                # Timeout if we haven't received first frame by self._first_frame_timeout sec
                if timeout is not None and waited >= timeout:
                    raise TimeoutError(f"No frame received on port {self._port} after {timeout:.0f}s. Is the headset streaming and pointed at this host?")
                
                if self._stall_warn and waited % self._stall_warn < poll_s:
                    print(f"[FrameSource] Waiting for frames ({waited:.0f}s)...")
            
            # If we get here, slot has been filled through _receive_loop()
            # Get the data in self._slot as well as the number of frames dropped before receiving this
            frame_id, left, right, metadata = self._slot
            dropped = self._dropped
            
            # Reset self._slot and self._dropped
            self._slot = None
            self._dropped = 0
        
        # Make sure self._awaiting_first is set to false as we're no longer waiting for the first frame
        self._awaiting_first = False
        
        # Return the frame
        return InputFrame(index=self._take_index(), left=left, right=right, metadata=metadata, frame_id=frame_id, dropped=dropped)
    
    def close(self):
        """TODO: docstring"""
        # Set stop
        self._stop.set()
        
        # Wake up all threads running on this condition
        with self._condition:
            self._condition.notify_all()
            
        # Wait until thread terminates
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        # Close socket
        self._socket.close(linger=0)
        # Terminate context
        self._context.term()
        
        print("[FrameSource] Socket cleanly closed.")

def make_frame_source(config_dict, target_extent) -> FrameSource:
    """Build the frame source named by `input_source`."""
    input_source = config_dict.get('input_source', 'directory')

    if input_source == 'directory':
        return DirectoryFrameSource(
            input=config_dict['input'],
            target_extent=target_extent,
            input_right=config_dict.get('input_right'),
            input_metadata=config_dict.get('input_metadata'),
        )

    if input_source == 'streamed':
        return StreamedFrameSource(
            target_extent=target_extent,
            port=config_dict.get('stream_port', 8099),
            connect=config_dict.get('stream_connect'),
        )

    raise ValueError(
        f"input_source {input_source} is not recognized. Please only use 'directory' or 'streamed'.")