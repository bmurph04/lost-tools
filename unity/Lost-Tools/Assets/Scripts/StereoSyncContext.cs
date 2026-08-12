using System;
using System.Text;
using Meta.XR;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using UnityEngine;

public class StereoSyncContext
{
    public int FrameId;
    public long TimestampUs;
    public long RightTimestampUs;
    
    // Left camera metadata
    public Pose LeftPose;
    public PassthroughCameraAccess.CameraIntrinsics LeftIntrinsics;
    public int LeftWidth, LeftHeight;
    public byte[] LeftJpegBytes = null;
    
    // Right camera metadata
    public Pose RightPose;
    public PassthroughCameraAccess.CameraIntrinsics RightIntrinsics;
    public int RightWidth, RightHeight;
    public byte[] RightJpegBytes = null;

    private PushSocket _pushSocket;
    private object _lock = new object();
    
    public StereoSyncContext(PushSocket socket)
    {
        _pushSocket = socket;
    }

    public void TrySendMultipart()
    {
        // Only one thread can run this block at a time, preventing desync sends
        lock (_lock)
        {
            // Return early if both images are not fully encoded
            if (_pushSocket == null || LeftJpegBytes == null || RightJpegBytes == null)
                return;

            try
            {
                // Initialize metadata to send
                var metadata = new
                {
                    frame_id = FrameId,
                    timestamp_us = TimestampUs,

                    leftCamera = new
                    {
                        timestamp_us = TimestampUs,
                        fx = LeftIntrinsics.FocalLength.x,
                        fy = LeftIntrinsics.FocalLength.y,
                        cx = LeftIntrinsics.PrincipalPoint.x,
                        cy = LeftIntrinsics.PrincipalPoint.y,
                        shape = new int[] { LeftHeight, LeftWidth },
                        pos = new float[] { LeftPose.position.x, LeftPose.position.y, LeftPose.position.z },
                        rot = new float[]
                            { LeftPose.rotation.x, LeftPose.rotation.y, LeftPose.rotation.z, LeftPose.rotation.w }
                    },

                    rightCamera = new
                    {
                        timestamp_us = RightTimestampUs,
                        fx = RightIntrinsics.FocalLength.x,
                        fy = RightIntrinsics.FocalLength.y,
                        cx = RightIntrinsics.PrincipalPoint.x,
                        cy = RightIntrinsics.PrincipalPoint.y,
                        shape = new int[] { RightHeight, RightWidth },
                        pos = new float[] { RightPose.position.x, RightPose.position.y, RightPose.position.z },
                        rot = new float[] { RightPose.rotation.x, RightPose.rotation.y, RightPose.rotation.z, RightPose.rotation.w }
                    }
                };
                
                string jsonMetadata = JsonConvert.SerializeObject(metadata);
                byte[] metaBytes = Encoding.UTF8.GetBytes(jsonMetadata);
            
                // Never block the Unity main thread if the receiver stalls
                if (!_pushSocket.TrySendMultipartBytes(TimeSpan.FromMilliseconds(20),
                        metaBytes, LeftJpegBytes, RightJpegBytes))
                {
                    Debug.LogWarning("[Streamer] Send timed out, dropping frame.");
                }
                
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[Streamer] Socket closed during active send: {exception.Message}");
            }
        } 
    }
}


