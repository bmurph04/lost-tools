using System;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Meta.XR;
using Meta.XR.EnvironmentDepth;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using Unity.Collections;
using Unity.Profiling.LowLevel.Unsafe;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;

public class StreamControllerUI : MonoBehaviour
{
    [Header("UI Component References")] [SerializeField]
    private TMP_InputField ipInputField;

    [SerializeField] private TMP_InputField portInputField;
    [SerializeField] private Toggle streamToggle;
    [SerializeField] private TMP_Text telemetryHUDText;

    [Header("Camera References")] [SerializeField]
    private PassthroughCameraAccess leftCameraAccess;

    [SerializeField] private PassthroughCameraAccess rightCameraAccess;
    // [SerializeField] private EnvironmentDepthManager  depthManager;

    [Header("Network References")] [SerializeField]
    private string targetIP;

    [SerializeField] private int targetPort;
    [SerializeField] private int targetSaveFPS;

    // One independent buffer set per in-flight capture, so concurrent
    // readbacks never write into memory another capture is still encoding.
    private const int BufferSlots = 2;
    private readonly NativeArray<byte>[] leftSlots = new NativeArray<byte>[BufferSlots];
    private readonly NativeArray<byte>[] rightSlots = new NativeArray<byte>[BufferSlots];
    private readonly bool[] slotBusy = new bool[BufferSlots];

    private bool isStreaming = false;
    private int capturesInFlight = 0;
    private int streamGeneration = 0;
    private bool closePending = false;
    
    private int frameCount = 0;
    private int renderFramesThisSecond = 0;
    private float currentRenderFPS = 0f;
    private int framesThisSecond;
    private float fpsTimer = 0f;
    private float currentFPS = 0f;
    private float nextSampleRealtime = 0f;

    private PushSocket pushSocket;

    void Start()
    {
        AsyncIO.ForceDotNet.Force();

        // Initialize IP and port if value associated
        if (ipInputField != null) ipInputField.text = targetIP;
        if (portInputField != null) portInputField.text = targetPort.ToString();

        // Initialize listeners for setting network parameters
        if (streamToggle != null) streamToggle.onValueChanged.AddListener(OnStreamToggleChanged);
        if (ipInputField != null) ipInputField.onEndEdit.AddListener(val => targetIP = val);
        if (portInputField != null) portInputField.onEndEdit.AddListener(val => int.TryParse(val, out targetPort));

        // Enable external managers
        // if (EnvironmentDepthManager.IsSupported) depthManager.enabled = true;
    }

    void Update()
    {
        CalculateFPS();
        UpdateTelemetryHUD();

        bool newCameraPair = leftCameraAccess != null && leftCameraAccess.IsUpdatedThisFrame &&
                             rightCameraAccess != null && rightCameraAccess.IsUpdatedThisFrame;

        // Debug.Log($"isStreaming = {isStreaming}; newCameraPair = {newCameraPair}");
        if (isStreaming && newCameraPair && capturesInFlight < BufferSlots)
        {
            bool sampleNow = ShouldSampleNow(targetSaveFPS);
            // Debug.Log($"ShouldSampleNow: {sampleNow}");
            if (sampleNow)
            {
                StreamLatestStereoFrameAsync();
            }
        }
        
        if (closePending && capturesInFlight == 0)
        {
            closePending = false;
            ClosePushSocket();
        }
    }

    /// <summary>
    /// Logic for when we should sample the next frame from the camera.
    /// </summary>
    /// <param name="fps">The target fps we should be sampling at.</param>
    /// <returns></returns>
    private bool ShouldSampleNow(int fps)
    {
        if (fps <= 0) return true;
        var now = Time.realtimeSinceStartup;
        if (now < nextSampleRealtime) return false;
        nextSampleRealtime = now + 1f / fps;
        return true;
    }

    /// <summary>
    /// Handles behavior when toggle is activated to start/stop streaming.
    /// </summary>
    /// <param name="isOn">The state of the streaming toggle.</param>
    private void OnStreamToggleChanged(bool isOn)
    {
        isStreaming = isOn;

        if (isStreaming)
        {
            if (ipInputField != null) ipInputField.interactable = false;
            if (portInputField != null) portInputField.interactable = false;

            // Connect socket when streaming starts
            pushSocket = new PushSocket();
            pushSocket.Options.Linger = TimeSpan.Zero;   // never block on unsent frames at close
            pushSocket.Options.SendHighWatermark = 4;    // bound the queue instead of buffering forever
            pushSocket.Connect($"tcp://{targetIP}:{targetPort}");
            Debug.Log($"[Streamer] Started streaming to {targetIP}:{targetPort}");
        }
        else
        {
            if (ipInputField != null) ipInputField.interactable = true;
            if (portInputField != null) portInputField.interactable = true;

            // Invalidate any capture already awaiting a GPU readback, then defer
            // teardown until it has actually finished touching the socket.
            streamGeneration++;
            closePending = true;
            Debug.Log("[Streamer] Stopping stream; socket close deferred.");
        }
    }

    /// <summary>
    /// Streams information from the latest frame over communication layer to be received by edge computer for
    /// backend memory module.
    /// </summary>
    private async void StreamLatestStereoFrameAsync()
    {
        int slot = AcquireSlot();
        if (slot < 0) return; // Every buffer set is busy, so skip this frame

        capturesInFlight++;

        try
        {
            await CaptureAndSendStereoFrameAsync(slot);
        }
        finally
        {
            capturesInFlight--;
            slotBusy[slot] = false;
        }
    }
    
    private async Awaitable CaptureAndSendStereoFrameAsync(int slot)
    {
        int generation = streamGeneration;
        int frameId = frameCount++;

        // Get current RGBA frame (GPU texture)
        Texture leftTex = leftCameraAccess.GetTexture();
        Texture rightTex = rightCameraAccess.GetTexture();
        
        // Return early if anything is null
        if (leftTex == null || rightTex == null || pushSocket == null) return;
        
        // Initialize or resize persistent memory buffers (Width * Height * 3 channels for RGB24)
        int leftLength = leftTex.width * leftTex.height * 3;
        int rightLength = rightTex.width * rightTex.height * 3;
        
        if (!leftSlots[slot].IsCreated || leftSlots[slot].Length != leftLength)
        {
            if (leftSlots[slot].IsCreated) leftSlots[slot].Dispose();
            leftSlots[slot] = new NativeArray<byte>(leftLength, Allocator.Persistent);
        }
        if (!rightSlots[slot].IsCreated || rightSlots[slot].Length != rightLength)
        {
            if (rightSlots[slot].IsCreated) rightSlots[slot].Dispose();
            rightSlots[slot] = new NativeArray<byte>(rightLength, Allocator.Persistent);
        }

        long leftTimestampUs = (leftCameraAccess.Timestamp.Ticks - DateTime.UnixEpoch.Ticks) / 10L;
        long rightTimestampUs = (rightCameraAccess.Timestamp.Ticks - DateTime.UnixEpoch.Ticks) / 10L;
        
        // Snapshot metadata for both cameras at this timestamp
        var syncContext = new StereoSyncContext(pushSocket)
        {
            FrameId = frameId,
            TimestampUs = leftTimestampUs,
            RightTimestampUs = rightTimestampUs,

            LeftPose = leftCameraAccess.GetCameraPose(),
            LeftIntrinsics = leftCameraAccess.Intrinsics,
            LeftWidth = leftTex.width,
            LeftHeight = leftTex.height,

            RightPose = rightCameraAccess.GetCameraPose(),
            RightIntrinsics = rightCameraAccess.Intrinsics,
            RightWidth = rightTex.width,
            RightHeight = rightTex.height
        };
        
        // Blit to RenderTextures to strip hardware stride padding (prevents diagonal static)
        RenderTexture leftRt = RenderTexture.GetTemporary(leftTex.width, leftTex.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        RenderTexture rightRt = RenderTexture.GetTemporary(rightTex.width, rightTex.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
    
        Graphics.Blit(leftTex, leftRt);
        Graphics.Blit(rightTex, rightRt);
        
        // Fire Async Requests (Queues parallel GPU copies directly into your persistent arrays) [cite: 876, 878]
        var leftReq = AsyncGPUReadback.RequestIntoNativeArrayAsync(ref leftSlots[slot], leftRt, 0, TextureFormat.RGB24);
        var rightReq = AsyncGPUReadback.RequestIntoNativeArrayAsync(ref rightSlots[slot], rightRt, 0, TextureFormat.RGB24);
        
        // Await both GPU transfers to finish without blocking the main thread [cite: 878]
        var leftResult = await leftReq;
        var rightResult = await rightReq;
        
        // Release RenderTextures back to memory pool
        RenderTexture.ReleaseTemporary(leftRt);
        RenderTexture.ReleaseTemporary(rightRt);

        // The socket may have been torn down while we were awaiting the readback
        if (generation != streamGeneration || pushSocket == null) return;

        if (leftResult.hasError || rightResult.hasError)
        {
            Debug.LogError("[Streamer] AsyncGPUReadback failed, dropping frame.");
            return;
        }

        // Encode the newly filled persistent buffers to JPG
        NativeArray<byte> leftJpgNative = ImageConversion.EncodeNativeArrayToJPG(
            leftSlots[slot], GraphicsFormat.R8G8B8_SRGB, (uint)syncContext.LeftWidth, (uint)syncContext.LeftHeight, rowBytes:0, quality:75);
    
        NativeArray<byte> rightJpgNative = ImageConversion.EncodeNativeArrayToJPG(
            rightSlots[slot], GraphicsFormat.R8G8B8_SRGB, (uint)syncContext.RightWidth, (uint)syncContext.RightHeight, rowBytes:0, quality:75);

        // Copy to managed arrays for ZeroMQ payload
        syncContext.LeftJpegBytes = leftJpgNative.ToArray();
        syncContext.RightJpegBytes = rightJpgNative.ToArray();

        // Prevent Unmanaged Memory Leaks
        leftJpgNative.Dispose();
        rightJpgNative.Dispose();

        // Send Frame
        syncContext.TrySendMultipart();
        
        framesThisSecond++;
    }

    private int AcquireSlot()
    {
        for (int i = 0; i < BufferSlots; i++)
        {
            if (!slotBusy[i])
            {
                slotBusy[i] = true;
                return i;
            }
        }

        return -1;
    }
    
    private void ClosePushSocket()
    {
        if (pushSocket == null) return;
        try
        {
            pushSocket.Close();
            pushSocket.Dispose();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Streamer] Error closing socket: {e.Message}");
        }
        finally
        {
            pushSocket = null;
        }
    }
    
    /// <summary>
    /// Calculates the FPS of the current frontend system.
    /// </summary>
    private void CalculateFPS()
    {
        renderFramesThisSecond++;
        fpsTimer += Time.deltaTime;
        if (fpsTimer >= 1.0f)
        {
            currentFPS = framesThisSecond / fpsTimer;
            currentRenderFPS = renderFramesThisSecond / fpsTimer;
            framesThisSecond = 0;
            renderFramesThisSecond = 0;
            fpsTimer = 0f;
        }
    }

    private void OnDestroy()
    {
        streamGeneration++;
        // Readbacks write directly into the persistent buffers; they must finish first
        AsyncGPUReadback.WaitAllRequests();

        ClosePushSocket();
        NetMQConfig.Cleanup(false);

        // Clean up persistent buffers
        for (int i = 0; i < BufferSlots; i++)
        {
            if (leftSlots[i].IsCreated) leftSlots[i].Dispose();
            if (rightSlots[i].IsCreated) rightSlots[i].Dispose();
        }
    }

    private void UpdateTelemetryHUD()
    {
        if (telemetryHUDText == null) return;
    
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("=== TELEMETRY HUD ===");
        sb.AppendLine($"Status: {(isStreaming ? "<color=green>STREAMING</color>" : "<color=red>IDLE</color>")}");
        sb.AppendLine($"Target: {targetIP}:{targetPort}");
    
        if (leftCameraAccess != null && leftCameraAccess.IsPlaying)
        {
            Vector2Int res = leftCameraAccess.CurrentResolution;
    
            sb.AppendLine($"Camera Eye: {leftCameraAccess.CameraPosition}");
            sb.AppendLine($"Render FPS: {currentRenderFPS:F1}");
            sb.AppendLine($"Stream FPS: {currentFPS:F1}");
            sb.AppendLine($"Resolution: {res.x} x {res.y}");
            sb.AppendLine($"Focal Length (fx, fy): ({leftCameraAccess.Intrinsics.FocalLength.x:F1}, {leftCameraAccess.Intrinsics.FocalLength.y:F1})");
        }
        else
        {
            sb.AppendLine("Camera Access: <color=yellow>Waiting for feed...</color>");
        }
    
        telemetryHUDText.text = sb.ToString();
    }
}