import { useEffect, useRef, useState } from "react";
import { Check, Pencil, X } from "lucide-react";
import { TranscriptUtterance } from "../types/meeting";

interface TranscriptPanelProps {
  transcriptLines: TranscriptUtterance[];
  sessionActive: boolean;
  onEditLine: (id: string, text: string) => void;
}

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

const SPEAKER_COLORS: Record<string, string> = {};
const PALETTE = [
  "bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300",
  "bg-violet-100 text-violet-800 dark:bg-violet-900/50 dark:text-violet-300",
  "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300",
  "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300",
  "bg-rose-100 text-rose-800 dark:bg-rose-900/50 dark:text-rose-300",
  "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/50 dark:text-cyan-300",
];
let paletteIndex = 0;

function speakerBadge(speaker: string) {
  if (speaker === "unknown") return "bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400";
  if (!SPEAKER_COLORS[speaker]) {
    SPEAKER_COLORS[speaker] = PALETTE[paletteIndex % PALETTE.length];
    paletteIndex += 1;
  }
  return SPEAKER_COLORS[speaker];
}

function UtteranceLine({
  line,
  onEdit,
}: {
  line: TranscriptUtterance;
  onEdit: (id: string, text: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(line.text);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);
  useEffect(() => { if (!editing) setDraft(line.text); }, [line.text, editing]);

  const save = () => {
    if (draft.trim()) onEdit(line.id, draft.trim());
    setEditing(false);
  };
  const cancel = () => { setDraft(line.text); setEditing(false); };

  return (
    <div className="group flex gap-3 py-3 px-4 border-b border-slate-100 dark:border-slate-800 last:border-0 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
      <div className="flex flex-col items-start gap-1 pt-0.5 shrink-0 w-[80px]">
        <span className={`inline-block px-1.5 py-0.5 rounded-full text-[10px] font-medium ${speakerBadge(line.speaker)}`}>
          {line.speaker === "unknown" ? "Speaker" : line.speaker.replace("SPEAKER_", "S")}
        </span>
        <span className="text-[10px] text-muted dark:text-slate-500 tabular-nums">
          {formatTime(line.start_time)}
        </span>
      </div>

      <div className="flex-1 min-w-0">
        {editing ? (
          <div className="space-y-2">
            <textarea
              ref={inputRef}
              className="w-full border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-2 text-sm text-ink dark:text-slate-100 bg-white dark:bg-slate-800 resize-none focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition"
              rows={2}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); save(); }
                if (e.key === "Escape") cancel();
              }}
            />
            <div className="flex gap-1.5">
              <button type="button" onClick={save}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-ink dark:bg-slate-100 text-white dark:text-slate-900 text-xs font-medium hover:bg-slate-700 dark:hover:bg-white transition">
                <Check size={11} /> Save
              </button>
              <button type="button" onClick={cancel}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 text-xs hover:bg-slate-50 dark:hover:bg-slate-700 transition">
                <X size={11} /> Cancel
              </button>
            </div>
          </div>
        ) : (
          <p className="text-sm text-ink dark:text-slate-200 leading-relaxed">{line.text}</p>
        )}
      </div>

      {!editing && (
        <button
          type="button"
          onClick={() => setEditing(true)}
          title="Edit this line"
          className="shrink-0 opacity-0 group-hover:opacity-100 mt-0.5 p-1 rounded-md text-muted dark:text-slate-500 hover:text-ink dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition"
        >
          <Pencil size={12} />
        </button>
      )}
    </div>
  );
}

export function TranscriptPanel({
  transcriptLines,
  sessionActive,
  onEditLine,
}: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  useEffect(() => {
    if (stickToBottomRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcriptLines]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-none px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-surface dark:bg-slate-900 sticky top-0 z-10">
        <h2 className="text-sm font-semibold text-ink dark:text-slate-100">Transcript</h2>
        <p className="text-xs text-muted dark:text-slate-500 mt-0.5">
          {transcriptLines.length > 0
            ? `${transcriptLines.length} line${transcriptLines.length !== 1 ? "s" : ""} · hover to edit`
            : sessionActive
            ? "Listening… first lines appear after the first chunk."
            : "Start a meeting to see the transcript."}
        </p>
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto bg-white dark:bg-slate-900"
      >
        {transcriptLines.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[200px]">
            <p className="text-sm text-muted dark:text-slate-500">
              {sessionActive ? "Waiting for audio…" : "No transcript yet."}
            </p>
          </div>
        ) : (
          transcriptLines.map((line) => (
            <UtteranceLine key={line.id} line={line} onEdit={onEditLine} />
          ))
        )}
      </div>
    </div>
  );
}
