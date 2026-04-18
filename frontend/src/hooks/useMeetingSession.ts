import { useEffect, useRef, useState } from "react";
import {
  ActionItem,
  BackendMessage,
  ChunkReceipt,
  Decision,
  ExportBundle,
  FrontendMessage,
  IntelligenceState,
  MeetingMetadata,
  MeetingSessionSnapshot,
  Risk,
  RollingMemoryStatus,
} from "../types/meeting";
import { useTranscript } from "./useTranscript";
import {
  createMeetingSocket,
  parseSocketMessage,
  sendSocketMessage,
} from "../utils/websocket";

const defaultSnapshot: MeetingSessionSnapshot = {
  meetingId: null,
  active: false,
  mode: null,
  startedAt: null,
  lastChunkSequence: 0,
  receivedChunks: 0,
};

const defaultRollingMemory: RollingMemoryStatus = {
  display_utterances: 0,
  llm_buffer_utterances: 0,
  previous_window_utterances: 0,
  asr_chunks_since_last_llm: 0,
  llm_window_ready: false,
  current_window_utterances: 0,
};

const defaultIntelligence: IntelligenceState = {
  summary: {
    running_summary: "",
    current_topic_focus: "",
    unresolved_issues: [],
    locked: false,
  },
  actionItems: [],
  decisions: [],
  risks: [],
  reviewFlags: [],
};

export function useMeetingSession() {
  const socketRef = useRef<WebSocket | null>(null);
  const sessionRef = useRef<MeetingSessionSnapshot>(defaultSnapshot);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const [connectionStatus, setConnectionStatus] = useState<
    "idle" | "connecting" | "connected" | "disconnected" | "error"
  >("idle");
  const [session, setSession] = useState<MeetingSessionSnapshot>(defaultSnapshot);
  const [receipts, setReceipts] = useState<ChunkReceipt[]>([]);
  const [messages, setMessages] = useState<string[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);
  const [rollingMemory, setRollingMemory] = useState<RollingMemoryStatus>(defaultRollingMemory);
  const [intelligence, setIntelligence] = useState<IntelligenceState>(defaultIntelligence);
  const [exportBundle, setExportBundle] = useState<ExportBundle | null>(null);
  const transcript = useTranscript();

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, []);

  const appendMessage = (message: string) => {
    setMessages((current) => {
      if (current[0] === message) return current;
      return [message, ...current].slice(0, 16);
    });
  };

  const scheduleReconnect = () => {
    if (!sessionRef.current.active || reconnectAttemptsRef.current >= 3) return;
    if (reconnectTimerRef.current) return;
    const delayMs = [1000, 2000, 5000][reconnectAttemptsRef.current] ?? 5000;
    reconnectAttemptsRef.current += 1;
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      connect(true);
    }, delayMs);
  };

  const connect = (isReconnect = false) => {
    if (
      socketRef.current &&
      (socketRef.current.readyState === WebSocket.OPEN ||
        socketRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    setConnectionStatus("connecting");
    if (!isReconnect) setLastError(null);

    const wsUrl =
      (import.meta.env.VITE_BACKEND_WS_URL as string | undefined) ??
      "ws://localhost:8000/ws/meeting";
    const socket = createMeetingSocket(wsUrl);

    socket.onopen = () => {
      reconnectAttemptsRef.current = 0;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      setConnectionStatus("connected");
      setLastError(null);
      if (sessionRef.current.active || sessionRef.current.meetingId) {
        socket.send(JSON.stringify({ type: "state_sync_request", payload: {} }));
      }
    };

    socket.onclose = () => {
      socketRef.current = null;
      setConnectionStatus("disconnected");
      scheduleReconnect();
    };

    socket.onerror = () => {
      setConnectionStatus("error");
      setLastError("Cannot reach the backend. Make sure it is running.");
    };

    socket.onmessage = (event) => {
      const parsed = parseSocketMessage(event.data);
      handleBackendMessage(parsed);
    };

    socketRef.current = socket;
  };

  const handleBackendMessage = (message: BackendMessage) => {
    switch (message.type) {
      case "session_created":
        transcript.reset();
        setReceipts([]);
        setRollingMemory(defaultRollingMemory);
        setIntelligence(defaultIntelligence);
        setExportBundle(null);
        setLastError(null);
        setSession({
          meetingId: message.payload.meeting_id,
          active: true,
          mode: message.payload.metadata.mode,
          startedAt: message.payload.created_at,
          lastChunkSequence: 0,
          receivedChunks: 0,
        });
        return;
      case "processing_started":
        return;
      case "transcript_update":
        transcript.appendUtterances(message.payload.utterances);
        setRollingMemory(message.payload.rolling_memory);
        return;
      case "speaker_backfill":
        transcript.applySpeakerBackfill(message.payload.updates);
        return;
      case "intelligence_update":
        setIntelligence({
          summary: message.payload.summary,
          actionItems: message.payload.action_items,
          decisions: message.payload.decisions,
          risks: message.payload.risks,
          reviewFlags: message.payload.review_flags,
        });
        return;
      case "trust_violation":
        appendMessage(message.payload.detail);
        return;
      case "processing_complete":
        setSession((current) => ({
          ...current,
          lastChunkSequence: message.payload.sequence_number,
          receivedChunks: current.receivedChunks + 1,
        }));
        setReceipts((current) =>
          [
            {
              sequenceNumber: message.payload.sequence_number,
              receivedAt: new Date().toISOString(),
              durationMs: message.payload.duration_ms,
              chunkBytes: message.payload.chunk_bytes,
              modelInfo: message.payload.processing_metadata.model_info,
              segmentCount: message.payload.processing_metadata.segment_count,
            },
            ...current,
          ].slice(0, 6)
        );
        return;
      case "state_sync":
        setSession(message.payload.session);
        transcript.reset();
        transcript.appendUtterances(message.payload.transcript ?? []);
        setRollingMemory(message.payload.rolling_memory ?? defaultRollingMemory);
        if (message.payload.intelligence) {
          setIntelligence({
            summary: message.payload.intelligence.summary,
            actionItems: message.payload.intelligence.action_items,
            decisions: message.payload.intelligence.decisions,
            risks: message.payload.intelligence.risks,
            reviewFlags: message.payload.intelligence.review_flags,
          });
        }
        if (message.payload.final_report) {
          setExportBundle({
            report: message.payload.final_report.report,
            jsonBase64: message.payload.final_report.json_base64,
            pdfBase64: message.payload.final_report.pdf_base64,
            storage: message.payload.final_report.storage,
          });
        }
        return;
      case "consolidation_complete":
        setExportBundle({
          report: message.payload.report,
          jsonBase64: message.payload.json_base64,
          pdfBase64: message.payload.pdf_base64,
          storage: message.payload.storage,
        });
        return;
      case "session_ended":
        setSession((current) => ({ ...current, active: false }));
        setRollingMemory(defaultRollingMemory);
        return;
      case "ack":
        return;
      case "error":
        setLastError(message.payload.detail);
        return;
      default:
        return;
    }
  };

  const sendMessage = (
    message: FrontendMessage,
    options?: { quiet?: boolean; autoReconnect?: boolean }
  ) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      if (options?.autoReconnect) connect(true);
      if (!options?.quiet) setLastError("Reconnecting. Please try again in a moment.");
      return false;
    }
    sendSocketMessage(socket, message);
    return true;
  };

  const startMeeting = (metadata: MeetingMetadata) =>
    new Promise<void>((resolve, reject) => {
      connect();
      const startedAt = new Date().toISOString();
      const timeoutAt = Date.now() + 10_000;
      let startMessageSent = false;

      const waitForSession = () => {
        const socket = socketRef.current;
        if (!socket) { reject(new Error("WebSocket was not created.")); return; }

        if (socket.readyState === WebSocket.OPEN && !sessionRef.current.meetingId && !startMessageSent) {
          startMessageSent = true;
          const sent = sendMessage({
            type: "meeting_start",
            payload: { metadata: { ...metadata, startedAt } },
          });
          if (!sent) { reject(new Error("Failed to send meeting_start.")); return; }
        }

        if (sessionRef.current.meetingId) { resolve(); return; }
        if (Date.now() >= timeoutAt) { reject(new Error("Meeting session timed out.")); return; }
        window.setTimeout(waitForSession, 150);
      };

      waitForSession();
    });

  const endMeeting = () => {
    if (!session.meetingId) return;
    sendMessage(
      { type: "meeting_end", payload: { meeting_id: session.meetingId } },
      { autoReconnect: true }
    );
  };

  const resetSession = () => {
    transcript.reset();
    setReceipts([]);
    setRollingMemory(defaultRollingMemory);
    setIntelligence(defaultIntelligence);
    setExportBundle(null);
    setLastError(null);
    setMessages([]);
    setSession(defaultSnapshot);
  };

  const editTranscriptLine = (id: string, text: string) => {
    transcript.editLine(id, text);
  };

  const updateSummary = (summary: string) => {
    if (!session.meetingId) return false;
    return sendMessage({
      type: "human_update_summary",
      payload: { meeting_id: session.meetingId, summary },
    } as FrontendMessage);
  };

  const addActionItem = (payload: {
    task: string;
    owner: string;
    deadline: string;
    priority: string;
    evidence: string;
  }) => {
    if (!session.meetingId) return false;
    return sendMessage({
      type: "human_add_action",
      payload: { meeting_id: session.meetingId, ...payload },
    } as FrontendMessage);
  };

  const addDecision = (payload: { decision: string; evidence: string }) => {
    if (!session.meetingId) return false;
    return sendMessage({
      type: "human_add_decision",
      payload: { meeting_id: session.meetingId, ...payload },
    } as FrontendMessage);
  };

  const addRisk = (payload: { risk: string; evidence: string }) => {
    if (!session.meetingId) return false;
    return sendMessage({
      type: "human_add_risk",
      payload: { meeting_id: session.meetingId, ...payload },
    } as FrontendMessage);
  };

  const updateItem = (
    itemType: "action_item" | "decision" | "risk",
    itemId: string,
    updates: Partial<ActionItem | Decision | Risk>
  ) => {
    if (!session.meetingId) return false;
    return sendMessage({
      type: "human_update_item",
      payload: {
        meeting_id: session.meetingId,
        item_type: itemType,
        item_id: itemId,
        updates,
      },
    } as FrontendMessage);
  };

  const deleteItem = (itemType: "action_item" | "decision" | "risk", itemId: string) => {
    if (!session.meetingId) return false;
    return sendMessage({
      type: "human_delete_item",
      payload: {
        meeting_id: session.meetingId,
        item_type: itemType,
        item_id: itemId,
      },
    } as FrontendMessage);
  };

  return {
    connectionStatus,
    session,
    receipts,
    messages,
    lastError,
    transcriptLines: transcript.lines,
    rollingMemory,
    intelligence,
    exportBundle,
    startMeeting,
    endMeeting,
    resetSession,
    editTranscriptLine,
    sendMessage,
    updateSummary,
    addActionItem,
    addDecision,
    addRisk,
    updateItem,
    deleteItem,
  };
}
