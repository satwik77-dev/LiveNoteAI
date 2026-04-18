export type FrontendMessageType =
  | "meeting_start"
  | "chunk_upload"
  | "meeting_end"
  | "human_add_action"
  | "human_add_decision"
  | "human_add_risk"
  | "human_update_item"
  | "human_update_action"
  | "human_delete_item"
  | "human_restore_item"
  | "human_update_summary"
  | "state_sync_request";

export type BackendMessageType =
  | "session_created"
  | "transcript_update"
  | "speaker_backfill"
  | "intelligence_update"
  | "processing_started"
  | "processing_complete"
  | "trust_violation"
  | "consolidation_complete"
  | "session_ended"
  | "state_sync"
  | "ack"
  | "error";

export interface MeetingMetadata {
  title: string;
  mode: "live" | "demo";
  startedAt?: string;
}

export interface ChunkPayload {
  meeting_id: string;
  sequence_number: number;
  mime_type: string;
  audio_base64: string;
  captured_at: string;
  duration_ms: number;
}

export interface MeetingStartMessage {
  type: "meeting_start";
  payload: {
    metadata: MeetingMetadata;
  };
}

export interface ChunkUploadMessage {
  type: "chunk_upload";
  payload: ChunkPayload;
}

export interface MeetingEndMessage {
  type: "meeting_end";
  payload: {
    meeting_id: string;
  };
}

export interface HumanUpdateSummaryMessage {
  type: "human_update_summary";
  payload: {
    meeting_id: string;
    summary: string;
  };
}

export interface HumanAddActionMessage {
  type: "human_add_action";
  payload: {
    meeting_id: string;
    task: string;
    owner: string;
    deadline: string;
    priority: string;
    evidence: string;
  };
}

export interface HumanUpdateItemMessage {
  type: "human_update_item" | "human_update_action";
  payload: {
    meeting_id: string;
    item_type: "action_item" | "decision" | "risk";
    item_id: string;
    updates: Record<string, unknown>;
  };
}

export interface HumanDeleteRestoreMessage {
  type: "human_delete_item" | "human_restore_item";
  payload: {
    meeting_id: string;
    item_type: "action_item" | "decision" | "risk";
    item_id: string;
  };
}

export interface HumanAddDecisionMessage {
  type: "human_add_decision";
  payload: {
    meeting_id: string;
    decision: string;
    evidence: string;
  };
}

export interface HumanAddRiskMessage {
  type: "human_add_risk";
  payload: {
    meeting_id: string;
    risk: string;
    evidence: string;
  };
}

export type FrontendMessage =
  | MeetingStartMessage
  | ChunkUploadMessage
  | MeetingEndMessage
  | HumanUpdateSummaryMessage
  | HumanAddActionMessage
  | HumanUpdateItemMessage
  | HumanDeleteRestoreMessage
  | HumanAddDecisionMessage
  | HumanAddRiskMessage;

export interface MeetingSessionSnapshot {
  meetingId: string | null;
  active: boolean;
  mode: "live" | "demo" | null;
  startedAt: string | null;
  lastChunkSequence: number;
  receivedChunks: number;
}

export interface ServerMessage<T extends BackendMessageType, P> {
  type: T;
  payload: P;
}

export interface TranscriptUtterance {
  id: string;
  chunk_id: number;
  speaker: string;
  text: string;
  start_time: number;
  end_time: number;
  confidence: number;
}

export interface RollingMemoryStatus {
  display_utterances: number;
  llm_buffer_utterances: number;
  previous_window_utterances: number;
  asr_chunks_since_last_llm: number;
  llm_window_ready: boolean;
  current_window_utterances: number;
}

export interface EvidenceSpan {
  utterance_id?: string | null;
  start_time: number;
  end_time: number;
  text: string;
}

export interface ActionItem {
  id: string;
  task: string;
  owner: string;
  deadline: string;
  normalized_deadline?: string | null;
  priority: string;
  status: string;
  needs_review: boolean;
  evidence: string;
  evidence_spans: EvidenceSpan[];
  chunk_origin: number;
  human_locked: boolean;
  deleted?: boolean;
}

export interface Decision {
  id: string;
  decision: string;
  evidence: string;
  evidence_spans: EvidenceSpan[];
  chunk_origin: number;
  human_locked: boolean;
  deleted?: boolean;
}

export interface Risk {
  id: string;
  risk: string;
  evidence: string;
  evidence_spans: EvidenceSpan[];
  chunk_origin: number;
  human_locked: boolean;
  deleted?: boolean;
}

export interface FinalReport {
  meeting: {
    meeting_id: string;
    title: string;
    mode: "live" | "demo";
    started_at: string | null;
    generated_at: string;
  };
  summary: IntelligenceState["summary"];
  action_items: ActionItem[];
  decisions: Decision[];
  risks: Risk[];
  review_flags: string[];
  transcript: TranscriptUtterance[];
}

export interface IntelligenceState {
  summary: {
    running_summary: string;
    current_topic_focus: string;
    unresolved_issues: string[];
    locked: boolean;
  };
  actionItems: ActionItem[];
  decisions: Decision[];
  risks: Risk[];
  reviewFlags: string[];
}

export type BackendMessage =
  | ServerMessage<
      "session_created",
      {
        meeting_id: string;
        created_at: string;
        metadata: MeetingMetadata;
      }
    >
  | ServerMessage<
      "processing_started",
      {
        meeting_id: string;
        sequence_number: number;
      }
    >
  | ServerMessage<
      "processing_complete",
      {
        meeting_id: string;
        sequence_number: number;
        chunk_bytes: number;
        mime_type: string;
        duration_ms: number;
        processing_metadata: {
          audio_duration_sec: number;
          sample_rate: number;
          channels: number;
          asr_time_ms: number;
          model_info: string;
          segment_count: number;
        };
      }
    >
  | ServerMessage<
      "transcript_update",
      {
        meeting_id: string;
        chunk_id: number;
        chunk_start_time: number;
        chunk_end_time: number;
        utterances: TranscriptUtterance[];
        diarization_pending: boolean;
        rolling_memory: RollingMemoryStatus;
      }
    >
  | ServerMessage<
      "speaker_backfill",
      {
        meeting_id: string;
        chunk_id: number;
        updates: Array<{
          utterance_id: string;
          speaker: string;
        }>;
        known_speakers: string[];
      }
    >
  | ServerMessage<
      "intelligence_update",
      {
        meeting_id: string;
        chunk_id: number;
        summary: IntelligenceState["summary"];
        action_items: ActionItem[];
        decisions: Decision[];
        risks: Risk[];
        review_flags: string[];
        llm_metadata: {
          provider: string;
          model: string;
          queued_windows: number;
        };
      }
    >
  | ServerMessage<
      "trust_violation",
      {
        meeting_id: string;
        chunk_id: number;
        rule: string;
        item_type: string;
        detail: string;
      }
    >
  | ServerMessage<
      "ack",
      {
        message: string;
        meeting_id?: string;
        sequence_number?: number;
      }
    >
  | ServerMessage<
      "state_sync",
      {
        session: MeetingSessionSnapshot;
        transcript: TranscriptUtterance[];
        rolling_memory: RollingMemoryStatus | null;
        intelligence: {
          meeting_id: string;
          chunk_id: number;
          summary: IntelligenceState["summary"];
          action_items: ActionItem[];
          decisions: Decision[];
          risks: Risk[];
          review_flags: string[];
          llm_metadata: {
            provider: string;
            model: string;
            queued_windows: number;
          };
        } | null;
        final_report: {
          report: FinalReport;
          json_base64: string;
          pdf_base64: string;
          storage?: {
            json_path: string;
            pdf_path: string;
          } | null;
        } | null;
      }
    >
  | ServerMessage<
      "consolidation_complete",
      {
        meeting_id: string;
        report: FinalReport;
        json_base64: string;
        pdf_base64: string;
        storage?: {
          json_path: string;
          pdf_path: string;
        } | null;
      }
    >
  | ServerMessage<
      "session_ended",
      {
        meeting_id: string;
        ended_at: string;
      }
    >
  | ServerMessage<
      "error",
      {
        code: string;
        detail: string;
      }
    >;

export interface ChunkReceipt {
  sequenceNumber: number;
  receivedAt: string;
  durationMs: number;
  chunkBytes: number;
  modelInfo?: string;
  segmentCount?: number;
}

export interface ExportBundle {
  report: FinalReport;
  jsonBase64: string;
  pdfBase64: string;
  storage?: {
    json_path: string;
    pdf_path: string;
  } | null;
}