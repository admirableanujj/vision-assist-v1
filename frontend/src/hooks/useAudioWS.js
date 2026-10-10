import { useState, useRef, useCallback } from 'react';

export function useAudioWS() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [genieResponse, setGenieResponse] = useState('');
  const [audioWsConnected, setAudioWsConnected] = useState(false);

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Initialize or connect WebSocket for audio
  const connectAudioWS = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return Promise.resolve(wsRef.current);
    }

    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setAudioWsConnected(true);
        resolve(ws);
      };

      ws.onclose = () => {
        setAudioWsConnected(false);
      };

      ws.onerror = (err) => {
        console.error("Audio WebSocket error:", err);
        reject(err);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.transcript) setTranscript(data.transcript);
          if (data.response) setGenieResponse(data.response);
        } catch (err) {
          console.error("Failed to parse audio response JSON:", err);
        }
      };
    });
  }, []);

  // Start Recording Audio (Push-to-Talk Press)
  const startListening = async () => {
    setTranscript('');
    setGenieResponse('');

    try {
      await connectAudioWS();

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
        }

        // Stop all microphone tracks to release hardware
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsListening(true);
    } catch (err) {
      console.error("Microphone access or recording error:", err);
      setIsListening(false);
    }
  };

  // Stop Recording Audio (Push-to-Talk Release)
  const stopListening = () => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
    }
  };

  return {
    isListening,
    transcript,
    genieResponse,
    audioWsConnected,
    startListening,
    stopListening,
  };
}