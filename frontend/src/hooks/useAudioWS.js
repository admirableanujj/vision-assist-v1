import { useState, useRef, useCallback, useEffect } from 'react';

export function useAudioWS() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [wsConnected, setWsConnected] = useState(false);

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Automatically establish WebSocket connection on mount
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
    
    console.log("Connecting Audio WebSocket to:", wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Audio WebSocket connected successfully!");
      setWsConnected(true);
    };

    ws.onclose = () => {
      console.log("Audio WebSocket closed");
      setWsConnected(false);
    };

    ws.onerror = (err) => {
      console.error("Audio WebSocket error:", err);
      setWsConnected(false);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.transcript) setTranscript(data.transcript);
        if (data.response) setAiResponse(data.response);
        if (data.message && !data.response) setAiResponse(data.message);
      } catch (err) {
        console.error("Failed to parse audio response JSON:", err);
      }
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  // Start Recording Audio (Push-to-Talk Press)
  const startRecording = async () => {
    setTranscript('');
    setAiResponse('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Send recorded audio blob over WebSocket to FastAPI backend
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(audioBlob);
        } else {
          console.warn("Audio WebSocket is not open. Trying to reconnect...");
          // Fallback or retry logic if needed
        }

        // Stop all microphone tracks to release hardware
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone access or recording error:", err);
      setIsRecording(false);
      setAiResponse("Microphone permission denied or hardware unavailable.");
    }
  };

  // Stop Recording Audio (Push-to-Talk Release)
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return {
    isRecording,
    transcript,
    aiResponse,
    wsConnected,
    startRecording,
    stopRecording,
    // Aliases for compatibility if needed elsewhere
    isListening: isRecording,
    genieResponse: aiResponse,
    audioWsConnected: wsConnected,
    startListening: startRecording,
    stopListening: stopRecording,
  };
}