import { useEffect, useRef, useState } from "react";
import { Mic, Moon, PlayCircle, Square, Sun, UploadCloud } from "lucide-react";
import { useMeetingSession } from "../hooks/useMeetingSession";
import { RecorderChunk, useMediaRecorder } from "../hooks/useMediaRecorder";
import { ActionItemsPanel } from "./ActionItemsPanel";
import { DecisionsPanel } from "./DecisionsPanel";
import { ExportPanel } from "./ExportPanel";
import { RisksPanel } from "./RisksPanel";
import { SummaryPanel } from "./SummaryPanel";
import { TranscriptPanel } from "./TranscriptPanel";

function formatElapsed(startedAt: string | null) {
  if (!startedAt) return "00:00";
  const secs = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  bytes.forEach((v) => { binary += String.fromCharCode(v); });
  return window.btoa(binary);
}

function DarkToggle({ dark, onToggle }: { dark: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      className="p-1.5 rounded-lg text-muted hover:text-ink dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-200 dark:hover:bg-slate-700 transition"
    >
      {dark ? <Sun size={15} /> : <Moon size={15} />}
    </button>
  );
}

export function MeetingDashboard() {
  const [meetingTitle, setMeetingTitle] = useState("Team Sync");
  const [meetingMode, setMeetingMode] = useState<"live" | "demo">("live");
  const [elapsed, setElapsed] = useState("00:00");
  const [demoFileUrl, setDemoFileUrl] = useState<string | null>(null);
  const [dark, setDark] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const meetingIdRef = useRef<string | null>(null);

  // ── Dark mode ────────────────────────────────────────────────────────────
  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggleDark = () => {
    const isDark = document.documentElement.classList.toggle("dark");
    setDark(isDark);
    localStorage.setItem("livenote-theme", isDark ? "dark" : "light");
  };

  // ── Session ──────────────────────────────────────────────────────────────
  const {
    connectionStatus,
    session,
    lastError,
    transcriptLines,
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
  } = useMeetingSession();

  useEffect(() => { meetingIdRef.current = session.meetingId; }, [session.meetingId]);

  const handleChunk = async (chunk: RecorderChunk) => {
    const mid = meetingIdRef.current;
    if (!mid) return;
    const audio_base64 = await blobToBase64(chunk.blob);
    sendMessage(
      {
        type: "chunk_upload",
        payload: {
          meeting_id: mid,
          sequence_number: chunk.sequenceNumber,
          mime_type: chunk.blob.type || "audio/webm",
          audio_base64,
          captured_at: chunk.createdAt,
          duration_ms: chunk.durationMs,
        },
      },
      { quiet: true, autoReconnect: true }
    );
  };

  const recorder = useMediaRecorder({ chunkMs: 15_000, onChunk: handleChunk });
  const isRecording = recorder.status === "recording";
  const hasSession = Boolean(session.meetingId);

  useEffect(() => {
    if (!session.active || !session.startedAt) { setElapsed("00:00"); return; }
    setElapsed(formatElapsed(session.startedAt));
    const interval = window.setInterval(() => setElapsed(formatElapsed(session.startedAt)), 1000);
    return () => window.clearInterval(interval);
  }, [session.active, session.startedAt]);

  const getDemoCaptureStream = () => {
    const audio = audioRef.current as HTMLAudioElement & {
      captureStream?: () => MediaStream;
      mozCaptureStream?: () => MediaStream;
    };
    return audio.captureStream?.() ?? audio.mozCaptureStream?.() ?? null;
  };

  const startMeetingFlow = async () => {
    await startMeeting({ title: meetingTitle, mode: meetingMode });
    if (meetingMode === "live") { await recorder.start(); return; }
    if (!audioRef.current || !demoFileUrl) return;
    const stream = getDemoCaptureStream();
    if (!stream) throw new Error("Demo audio capture not supported in this browser.");
    recorder.prepareFromStream(stream);
    await recorder.start();
    await audioRef.current.play();
  };

  const stopMeeting = () => {
    audioRef.current?.pause();
    recorder.stop();
    endMeeting();
  };

  // ── Shared header pieces ─────────────────────────────────────────────────
  const statusPill = (
    <span
      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
        connectionStatus === "connected"
          ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400"
          : connectionStatus === "connecting"
          ? "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-400"
          : "bg-slate-200 text-slate-500 dark:bg-slate-700 dark:text-slate-400"
      }`}
    >
      {connectionStatus === "connected"
        ? "Connected"
        : connectionStatus === "connecting"
        ? "Connecting…"
        : "Not connected"}
    </span>
  );

  // ── Pre-meeting ───────────────────────────────────────────────────────────
  if (!hasSession) {
    return (
      <div className="min-h-screen bg-bg dark:bg-slate-950 flex flex-col transition-colors">
        <header className="border-b border-slate-200 dark:border-slate-800 bg-surface dark:bg-slate-900 px-6 py-3 flex items-center justify-between">
          <span className="text-sm font-semibold text-ink dark:text-slate-100 tracking-wide">LiveNote</span>
          <div className="flex items-center gap-2">
            {statusPill}
            <DarkToggle dark={dark} onToggle={toggleDark} />
          </div>
        </header>

        <div className="flex-1 flex items-center justify-center px-4 py-12">
          <div className="w-full max-w-md">
            <h1 className="text-2xl font-semibold text-ink dark:text-slate-100 mb-1">
              Start a meeting
            </h1>
            <p className="text-sm text-muted dark:text-slate-400 mb-6 leading-relaxed">
              LiveNote captures audio in real time and builds a live transcript, summary,
              action items, decisions, and risks as you speak.
            </p>

            <div className="bg-surface dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-card space-y-4">
              <div>
                <label className="block text-xs font-medium text-muted dark:text-slate-400 uppercase tracking-wide mb-1.5">
                  Meeting title
                </label>
                <input
                  className="w-full border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-2.5 text-sm text-ink dark:text-slate-100 bg-white dark:bg-slate-800 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition"
                  value={meetingTitle}
                  onChange={(e) => setMeetingTitle(e.target.value)}
                  placeholder="Weekly team sync"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-muted dark:text-slate-400 uppercase tracking-wide mb-1.5">
                  Audio source
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {(["live", "demo"] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setMeetingMode(mode)}
                      className={`flex items-center justify-center gap-2 rounded-lg border py-2.5 text-sm font-medium transition ${
                        meetingMode === mode
                          ? "border-accent bg-accent/10 text-accent dark:bg-accent/20"
                          : "border-slate-300 dark:border-slate-600 text-muted dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-500 hover:text-ink dark:hover:text-slate-200"
                      }`}
                    >
                      {mode === "live" ? <Mic size={14} /> : <PlayCircle size={14} />}
                      {mode === "live" ? "Live microphone" : "Demo audio"}
                    </button>
                  ))}
                </div>
              </div>

              {meetingMode === "demo" && (
                <div>
                  <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-slate-300 dark:border-slate-600 px-3 py-2.5 text-sm text-muted dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-500 hover:text-ink dark:hover:text-slate-200 transition">
                    <UploadCloud size={14} />
                    <span>{demoFileUrl ? "Replace audio file" : "Choose a prerecorded audio file"}</span>
                    <input
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (!file) return;
                        if (demoFileUrl) URL.revokeObjectURL(demoFileUrl);
                        setDemoFileUrl(URL.createObjectURL(file));
                      }}
                    />
                  </label>
                  {demoFileUrl && (
                    <audio
                      ref={audioRef}
                      className="mt-3 w-full h-9"
                      controls
                      src={demoFileUrl}
                      onEnded={() => { if (session.active) stopMeeting(); }}
                    />
                  )}
                </div>
              )}

              {lastError && (
                <p className="text-xs text-recording bg-red-50 dark:bg-red-900/30 border border-red-100 dark:border-red-800 rounded-lg px-3 py-2">
                  {lastError}
                </p>
              )}

              <button
                type="button"
                disabled={meetingMode === "demo" && !demoFileUrl}
                onClick={startMeetingFlow}
                className="w-full bg-ink dark:bg-slate-100 text-white dark:text-slate-900 rounded-lg py-2.5 text-sm font-medium hover:bg-slate-700 dark:hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
              >
                {meetingMode === "live" ? <Mic size={14} /> : <PlayCircle size={14} />}
                Start meeting
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Active / post-meeting layout ──────────────────────────────────────────
  return (
    <div className="h-screen flex flex-col bg-bg dark:bg-slate-950 overflow-hidden transition-colors">
      {/* Header */}
      <header className="flex-none border-b border-slate-200 dark:border-slate-800 bg-surface dark:bg-slate-900 px-5 h-[52px] flex items-center gap-3">
        <span className="text-sm font-semibold text-ink dark:text-slate-100 shrink-0">LiveNote</span>
        <span className="h-4 w-px bg-slate-200 dark:bg-slate-700 shrink-0" />
        <span className="text-sm text-ink dark:text-slate-200 font-medium truncate flex-1 min-w-0">
          {meetingTitle}
        </span>

        <div className="flex items-center gap-2.5 shrink-0 ml-auto">
          {session.active && (
            <>
              {isRecording ? (
                <span className="flex items-center gap-1.5 text-xs font-medium text-recording">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-recording opacity-60" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-recording" />
                  </span>
                  Recording
                </span>
              ) : (
                <span className="text-xs text-muted dark:text-slate-500">Standby</span>
              )}
              <span className="text-sm font-mono font-medium text-ink dark:text-slate-200 tabular-nums w-12 text-right">
                {elapsed}
              </span>
              <button
                type="button"
                onClick={stopMeeting}
                className="flex items-center gap-1.5 rounded-lg bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 px-3 py-1.5 text-xs font-medium text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/50 transition"
              >
                <Square size={11} fill="currentColor" />
                End meeting
              </button>
            </>
          )}

          {!session.active && (
            <>
              <span className="text-xs text-muted dark:text-slate-500">Ended · {elapsed}</span>
              <button
                type="button"
                onClick={resetSession}
                className="rounded-lg border border-slate-200 dark:border-slate-700 bg-surface dark:bg-slate-800 px-3 py-1.5 text-xs font-medium text-ink dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition"
              >
                New meeting
              </button>
            </>
          )}

          <div className="h-4 w-px bg-slate-200 dark:bg-slate-700" />
          <DarkToggle dark={dark} onToggle={toggleDark} />
        </div>
      </header>

      {lastError && (
        <div className="flex-none bg-red-50 dark:bg-red-900/30 border-b border-red-100 dark:border-red-800 px-5 py-2 text-xs text-red-600 dark:text-red-400">
          {lastError}
        </div>
      )}

      {/* Body — transcript sidebar | intelligence main panel */}
      <div className="flex-1 min-h-0 lg:grid lg:grid-cols-[380px_1fr]">
        {/* Left: Transcript sidebar */}
        <div className="overflow-y-auto border-r border-slate-200 dark:border-slate-800 h-full">
          <TranscriptPanel
            transcriptLines={transcriptLines}
            sessionActive={session.active}
            onEditLine={editTranscriptLine}
          />
        </div>

        {/* Right: Intelligence main area */}
        <div className="overflow-y-auto h-full bg-slate-100/50 dark:bg-slate-950">
          <div className="p-5 space-y-4">
            <SummaryPanel
              runningSummary={intelligence.summary.running_summary}
              currentTopicFocus={intelligence.summary.current_topic_focus}
              unresolvedIssues={intelligence.summary.unresolved_issues}
              locked={intelligence.summary.locked}
              onSave={updateSummary}
            />
            <ActionItemsPanel
              actionItems={intelligence.actionItems}
              onAdd={addActionItem}
              onUpdate={(id, updates) => updateItem("action_item", id, updates)}
              onDelete={(id) => deleteItem("action_item", id)}
            />
            {/* Decisions + Risks side by side on wide screens */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <DecisionsPanel
                decisions={intelligence.decisions}
                onAdd={addDecision}
                onUpdate={(id, updates) => updateItem("decision", id, updates)}
                onDelete={(id) => deleteItem("decision", id)}
              />
              <RisksPanel
                risks={intelligence.risks}
                onAdd={addRisk}
                onUpdate={(id, updates) => updateItem("risk", id, updates)}
                onDelete={(id) => deleteItem("risk", id)}
              />
            </div>
            <ExportPanel exportBundle={exportBundle} sessionActive={session.active} />
          </div>
        </div>
      </div>
    </div>
  );
}
