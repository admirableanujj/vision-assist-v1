// frontend/src/hooks/useVisionWS.js
import { useState, useEffect, useRef, useCallback } from 'react';

export function useVisionWS(isStreamActive = true, zoneId = 'zone-1') {
  const [wsConnected, setWsConnected] = useState(false);
  const [detections, setDetections] = useState([]);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  // Initialize Webcam Stream
  useEffect(() => {
    if (!isStreamActive) return;

    let activeStream = null;
    navigator.mediaDevices.getUserMedia({ 
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } } 
    })
      .then((stream) => {
        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => console.error("Webcam access error:", err));

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isStreamActive]);

  // Initialize WebSocket Connection & Capture Loop (Re-runs if stream active state or zone changes)
  useEffect(() => {
    if (!isStreamActive) return;

    // Determine WebSocket protocol and append active zone query parameter
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/vision?zone=${zoneId}`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`Vision WebSocket connected for zone: ${zoneId}`);
      setWsConnected(true);
    };

    ws.onclose = () => {
      console.log("Vision WebSocket disconnected.");
      setWsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("Vision WebSocket error:", error);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data && data.detections) {
          setDetections(data.detections);
        }
      } catch (err) {
        console.error("Failed to parse WebSocket message:", err);
      }
    };

    // Frame Capture Loop (Sends snapshot every 100ms ~ 10 FPS)
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    intervalRef.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN && videoRef.current) {
        const video = videoRef.current;
        if (video.videoWidth === 0 || video.videoHeight === 0) return;

        canvas.width = 640;
        canvas.height = 480;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) {
            ws.send(blob);
          }
        }, "image/jpeg", 0.65);
      }
    }, 100);

    return () => {
      clearInterval(intervalRef.current);
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, [isStreamActive, zoneId]);

  // Render bounding boxes over HTML5 canvas overlay whenever detections update
  const renderCanvasOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detections.length) return;

    // Match canvas internal resolution to video display bounds
    const displayWidth = video.clientWidth || 640;
    const displayHeight = video.clientHeight || 480;
    
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    const scaleX = displayWidth / 640;
    const scaleY = displayHeight / 480;

    detections.forEach(({ label, confidence, box }) => {
      let [x1, y1, x2, y2] = box;
      
      // Scale coordinates to match current canvas viewport
      x1 *= scaleX;
      y1 *= scaleY;
      x2 *= scaleX;
      y2 *= scaleY;

      const width = x2 - x1;
      const height = y2 - y1;

      // Draw bounding box
      ctx.strokeStyle = "#10B981"; // Emerald green
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, width, height);

      // Draw background tag pill (Fixed confidence multiplication typo here)
      const labelText = `${label} ${(confidence * 100).toFixed(0)}%`;
      ctx.font = "12px Inter, system-ui, sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const textHeight = 16;
      
      ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
      ctx.fillRect(x1, y1 > textHeight ? y1 - textHeight : y1, textMetrics.width + 10, textHeight);

      // Draw label text
      ctx.fillStyle = "#020617"; // Slate 950
      ctx.fillText(labelText, x1 + 5, y1 > textHeight ? y1 - 4 : y1 + 12);
    });
  }, [detections]);

  useEffect(() => {
    renderCanvasOverlay();
  }, [detections, renderCanvasOverlay]);

  return {
    videoRef,
    canvasRef,
    wsConnected,
    detections,
  };
}