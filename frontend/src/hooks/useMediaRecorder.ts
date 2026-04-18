import { useEffect, useRef, useState } from "react";

export interface RecorderChunk {
  blob: Blob;
  sequenceNumber: number;
  createdAt: string;
  durationMs: number;
}

export interface MediaRecorderState {
  permission: "idle" | "granted" | "denied";
  status: "idle" | "requesting" | "ready" | "recording" | "error";
  error: string | null;
}

interface UseMediaRecorderOptions {
  chunkMs?: number;
  onChunk?: (chunk: RecorderChunk) => void;
}

export function useMediaRecorder({
  chunkMs = 15_000,
  onChunk,
}: UseMediaRecorderOptions = {}) {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onChunkRef = useRef<typeof onChunk>(onChunk);
  const chunkStartedAtRef = useRef<number | null>(null);
  const sequenceRef = useRef(0);
  const restartTimeoutRef = useRef<number | null>(null);
  const shouldContinueRef = useRef(false);
  const usingExternalStreamRef = useRef(false);
  const [state, setState] = useState<MediaRecorderState>({
    permission: "idle",
    status: "idle",
    error: null,
  });

  useEffect(() => {
    onChunkRef.current = onChunk;
  }, [onChunk]);

  useEffect(() => {
    return () => {
      if (restartTimeoutRef.current) {
        window.clearTimeout(restartTimeoutRef.current);
      }
      recorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const scheduleStop = () => {
    if (restartTimeoutRef.current) {
      window.clearTimeout(restartTimeoutRef.current);
    }

    restartTimeoutRef.current = window.setTimeout(() => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state === "recording") {
        recorder.stop();
      }
    }, chunkMs);
  };

  const createRecorder = (stream: MediaStream) => {
    const recorder = new MediaRecorder(stream, {
      mimeType: "audio/webm;codecs=opus",
    });

    recorder.ondataavailable = (event) => {
      if (!event.data || event.data.size === 0) {
        return;
      }

      sequenceRef.current += 1;
      const startedAt = chunkStartedAtRef.current ?? Date.now() - chunkMs;
      const durationMs = Date.now() - startedAt;

      onChunkRef.current?.({
        blob: event.data,
        sequenceNumber: sequenceRef.current,
        createdAt: new Date().toISOString(),
        durationMs,
      });

      chunkStartedAtRef.current = Date.now();
    };

    recorder.onstop = () => {
      if (!shouldContinueRef.current) {
        return;
      }

      const activeStream = streamRef.current;
      if (!activeStream) {
        return;
      }

      const nextRecorder = createRecorder(activeStream);
      recorderRef.current = nextRecorder;
      chunkStartedAtRef.current = Date.now();
      nextRecorder.start();
      scheduleStop();
    };

    recorder.onerror = () => {
      setState((current) => ({
        ...current,
        status: "error",
        error: "MediaRecorder encountered an error.",
      }));
    };

    return recorder;
  };

  const prepare = async () => {
    try {
      setState({
        permission: "idle",
        status: "requesting",
        error: null,
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
      });
      usingExternalStreamRef.current = false;
      streamRef.current = stream;
      recorderRef.current = createRecorder(stream);
      sequenceRef.current = 0;
      setState({
        permission: "granted",
        status: "ready",
        error: null,
      });
    } catch {
      setState({
        permission: "denied",
        status: "error",
        error: "Microphone permission denied or unavailable.",
      });
    }
  };

  const prepareFromStream = (stream: MediaStream) => {
    usingExternalStreamRef.current = true;
    streamRef.current = stream;
    recorderRef.current = createRecorder(stream);
    sequenceRef.current = 0;
    setState({
      permission: "granted",
      status: "ready",
      error: null,
    });
  };

  const start = async () => {
    if (!recorderRef.current) {
      await prepare();
    }

    const recorder = recorderRef.current;
    if (!recorder) {
      return;
    }

    if (recorder.state === "recording") {
      return;
    }

    shouldContinueRef.current = true;
    chunkStartedAtRef.current = Date.now();
    recorder.start();
    scheduleStop();
    setState((current) => ({
      ...current,
      status: "recording",
    }));
  };

  const stop = () => {
    shouldContinueRef.current = false;
    if (restartTimeoutRef.current) {
      window.clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }
    const recorder = recorderRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }

    if (!usingExternalStreamRef.current) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    }
    streamRef.current = null;
    recorderRef.current = null;
    chunkStartedAtRef.current = null;
    usingExternalStreamRef.current = false;

    setState((current) => ({
      ...current,
      status: "idle",
    }));
  };

  return {
    ...state,
    prepare,
    prepareFromStream,
    start,
    stop,
  };
}
