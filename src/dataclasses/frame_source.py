"""
Frame sources.

The pipeline is already causal and frame-at-a-time, so replaying a directory of
captured frames and consuming a live headset stream differ only in where a frame
comes from. Both sources yield the same `StreamFrame`, so the main loop body is
identical either way and directory replay stays available for debugging.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.utils import load_frame, load_serialized_data, frames_sort_key


@dataclass
class StreamFrame:
    """One frame of input, however it was obtained."""
    index: int                      # pipeline clock; contiguous even when the source drops
    left: np.ndarray                # (3, H, W) uint8 RGB at the target extent
    right: Optional[np.ndarray]     # (3, H, W) uint8 RGB at the target extent, or None
    metadata: Optional[Dict]        # per-frame capture metadata, or None
    frame_id: Optional[int] = None  # id assigned by the source (the headset frame_id)
    dropped: int = 0                # source frames skipped since the previous delivered frame


class FrameSource:
    """
    Base for frame sources.

    `bootstrap()` returns the first frame and must be called before iterating.
    Geometry setup needs frame 0's metadata and native resolution, and a live
    source cannot know either until something arrives. The bootstrapped frame is
    held and re-delivered as the first iteration, so setup consumes no frames.
    """

    def __init__(self, target_extent):
        self.target_extent = target_extent  # (height, width)
        self.source_size = None             # (width, height) as received, before resizing
        self._pending: Optional[StreamFrame] = None
        self._index = 0

    # -- lifecycle ----------------------------------------------------------

    def bootstrap(self) -> StreamFrame:
        if self._pending is None:
            self._pending = self._next_frame()
        return self._pending

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- iteration ----------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> StreamFrame:
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        return self._next_frame()

    def _next_frame(self) -> StreamFrame:
        raise NotImplementedError

    def _take_index(self) -> int:
        index = self._index
        self._index += 1
        return index

    # -- shared helpers -----------------------------------------------------

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
    """Replay of a captured sequence. The offline path, unchanged in behaviour."""

    def __init__(self, input_dir, target_extent, input_right=None, input_metadata=None):
        super().__init__(target_extent)

        self._left_paths = self._sorted_dir(input_dir)
        if not self._left_paths:
            raise RuntimeError(f"No frames found in {input_dir}")

        self._right_paths = self._sorted_dir(input_right) if input_right is not None else None
        self._metadata_paths = self._sorted_dir(input_metadata) if input_metadata is not None else None

        if self._right_paths is not None:
            assert len(self._right_paths) == len(self._left_paths), \
                f"Sequence length mismatch, left frames_dir has {len(self._left_paths)} frames " \
                f"but right_frames_dir has {len(self._right_paths)} frames"
        if self._metadata_paths is not None:
            assert len(self._metadata_paths) == len(self._left_paths), \
                f"Sequence length mismatch, left frames_dir has {len(self._left_paths)} frames " \
                f"but metadata_dir has {len(self._metadata_paths)} frames"

        # Native resolution, read once, so the intrinsics scale factor is
        # computed against the size the camera actually produced.
        _, source_height, source_width = load_frame(str(self._left_paths[0])).shape
        self.source_size = (source_width, source_height)

    @staticmethod
    def _sorted_dir(directory):
        return sorted([f for f in Path(directory).iterdir()], key=frames_sort_key)

    def __len__(self):
        return len(self._left_paths)

    def _next_frame(self) -> StreamFrame:
        i = self._index
        if i >= len(self._left_paths):
            raise StopIteration

        height, width = self.target_extent
        left = load_frame(str(self._left_paths[i]), extent=(height, width))
        right = load_frame(str(self._right_paths[i]), extent=(height, width)) \
            if self._right_paths is not None else None
        metadata = load_serialized_data(str(self._metadata_paths[i])) \
            if self._metadata_paths is not None else None

        return StreamFrame(index=self._take_index(), left=left, right=right,
                           metadata=metadata, frame_id=i)


class ZMQFrameSource(FrameSource):
    """
    Live headset stream.

    A receiver thread owns the socket and keeps only the newest decoded frame in
    a one-slot buffer. Falling behind therefore costs frames, not freshness --
    latency never grows with how far behind the consumer is, which is what a
    plain blocking `recv_multipart` gets wrong. ZMQ_CONFLATE would be the obvious
    alternative but does not support multipart messages, and the headset sends
    (metadata, left, right) as three parts.

    JPEG decode and resize happen on the receiver thread so the pipeline thread
    never pays for a frame it is about to drop.
    """

    def __init__(self, target_extent, port=8099, connect=None, rcvhwm=10,
                 first_frame_timeout=60.0, stall_warn_s=5.0):
        super().__init__(target_extent)
        import zmq  # imported here so the offline path does not require pyzmq

        self._zmq = zmq
        self._port = port
        self._stall_warn_s = stall_warn_s
        self._first_frame_timeout = first_frame_timeout
        self._awaiting_first = True

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        # Bounded so the socket cannot hoard frames behind our one-slot buffer;
        # the receiver thread drains far faster than the headset sends.
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

        self._thread = threading.Thread(target=self._receive_loop, name="frame-receiver", daemon=True)
        self._thread.start()
        print(f"[FrameSource] Listening for headset stream on port {port}...")

    # -- receiver thread ----------------------------------------------------

    def _receive_loop(self):
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        while not self._stop.is_set():
            try:
                # Polling rather than blocking keeps the thread responsive to close().
                if not poller.poll(200):
                    continue
                parts = self._socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                continue
            except (zmq.ContextTerminated, zmq.ZMQError):
                break  # socket torn down under us during shutdown

            decoded = self._decode(parts)
            if decoded is None:
                continue

            with self._condition:
                if self._slot is not None:
                    # Pipeline is behind: keep the newer frame, drop the older.
                    self._dropped += 1
                self._slot = decoded
                self._condition.notify()

    def _decode(self, parts):
        if len(parts) != 3:
            self.decode_failures += 1
            return None

        metadata_bytes, left_bytes, right_bytes = parts
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            frame_id = int(metadata["frame_id"])
        except (ValueError, KeyError, UnicodeDecodeError) as error:
            print(f"[FrameSource] Malformed metadata, skipping frame: {error}")
            self.decode_failures += 1
            return None

        # Out-of-order arrival: a newer frame has already gone through.
        if frame_id <= self._latest_frame_id:
            return None

        left_bgr = cv2.imdecode(np.frombuffer(left_bytes, np.uint8), cv2.IMREAD_COLOR)
        right_bgr = cv2.imdecode(np.frombuffer(right_bytes, np.uint8), cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None:
            print("[FrameSource] Received invalid or corrupted image bytes, skipping frame")
            self.decode_failures += 1
            return None

        if self.source_size is None:
            self.source_size = (left_bgr.shape[1], left_bgr.shape[0])

        self._latest_frame_id = frame_id
        return (frame_id, self._to_chw_rgb(left_bgr), self._to_chw_rgb(right_bgr), metadata)

    # -- pipeline thread ----------------------------------------------------

    def _next_frame(self) -> StreamFrame:
        timeout = self._first_frame_timeout if self._awaiting_first else None
        waited = 0.0
        poll_s = 0.5

        with self._condition:
            while self._slot is None:
                if self._stop.is_set():
                    raise StopIteration
                # Short waits rather than one long one, so Ctrl-C stays responsive.
                self._condition.wait(timeout=poll_s)
                if self._slot is not None:
                    break
                waited += poll_s
                if timeout is not None and waited >= timeout:
                    raise TimeoutError(
                        f"No frame received on port {self._port} after {timeout:.0f}s. "
                        f"Is the headset streaming and pointed at this host?")
                if self._stall_warn_s and waited % self._stall_warn_s < poll_s:
                    print(f"[FrameSource] Waiting for frames ({waited:.0f}s)...")

            frame_id, left, right, metadata = self._slot
            dropped = self._dropped
            self._slot = None
            self._dropped = 0

        self._awaiting_first = False
        return StreamFrame(index=self._take_index(), left=left, right=right,
                           metadata=metadata, frame_id=frame_id, dropped=dropped)

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._socket.close(linger=0)
        self._context.term()
        print("[FrameSource] Socket cleanly closed.")


def make_frame_source(config_dict, target_extent) -> FrameSource:
    """Build the frame source named by `input_source`."""
    input_source = config_dict.get('input_source', 'directory')

    if input_source == 'directory':
        return DirectoryFrameSource(
            input_dir=config_dict['input'],
            target_extent=target_extent,
            input_right=config_dict.get('input_right'),
            input_metadata=config_dict.get('input_metadata'),
        )

    if input_source == 'zmq':
        return ZMQFrameSource(
            target_extent=target_extent,
            port=config_dict.get('stream_port', 8099),
            connect=config_dict.get('stream_connect'),
        )

    raise ValueError(
        f"input_source {input_source} is not recognized. Please only use 'directory' or 'zmq'.")