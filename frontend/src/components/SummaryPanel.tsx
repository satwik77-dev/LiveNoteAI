import { useEffect, useState } from "react";
import { Check, Lock, Pencil, X } from "lucide-react";

interface SummaryPanelProps {
  runningSummary: string;
  currentTopicFocus: string;
  unresolvedIssues: string[];
  locked: boolean;
  onSave: (summary: string) => void;
}

export function SummaryPanel({
  runningSummary,
  currentTopicFocus,
  unresolvedIssues,
  locked,
  onSave,
}: SummaryPanelProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(runningSummary);

  useEffect(() => { if (!editing) setDraft(runningSummary); }, [runningSummary, editing]);

  const save = () => { onSave(draft); setEditing(false); };
  const cancel = () => { setDraft(runningSummary); setEditing(false); };

  return (
    <section className="bg-surface dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden shadow-card">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-100 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-ink dark:text-slate-100">Summary</h3>
          {locked && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400 text-[11px] font-medium">
              <Lock size={9} /> Locked
            </span>
          )}
        </div>
        {!editing ? (
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex items-center gap-1.5 text-xs text-muted dark:text-slate-400 hover:text-ink dark:hover:text-slate-200 transition"
          >
            <Pencil size={12} /> Edit
          </button>
        ) : (
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={save}
              className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-ink dark:bg-slate-100 text-white dark:text-slate-900 text-xs font-medium hover:bg-slate-700 dark:hover:bg-white transition">
              <Check size={11} /> Save
            </button>
            <button type="button" onClick={cancel}
              className="flex items-center gap-1 px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 text-xs hover:bg-slate-50 dark:hover:bg-slate-700 transition">
              <X size={11} /> Cancel
            </button>
          </div>
        )}
      </div>

      <div className="p-5 space-y-4">
        {editing ? (
          <textarea
            autoFocus
            className="w-full border border-slate-200 dark:border-slate-600 rounded-lg px-3 py-2.5 text-sm text-ink dark:text-slate-100 bg-white dark:bg-slate-800 resize-none focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition"
            rows={6}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Edit the running summary…"
          />
        ) : (
          <p className="text-sm text-ink dark:text-slate-200 leading-relaxed min-h-[56px] whitespace-pre-wrap">
            {runningSummary || (
              <span className="text-muted dark:text-slate-500">
                LiveNote will build the summary here as the conversation progresses.
              </span>
            )}
          </p>
        )}

        {(currentTopicFocus || unresolvedIssues.length > 0) && !editing && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            {currentTopicFocus && (
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-100 dark:border-slate-700 px-4 py-3">
                <p className="text-[10px] font-semibold text-muted dark:text-slate-500 uppercase tracking-wider mb-1.5">
                  Current topic
                </p>
                <p className="text-xs text-ink dark:text-slate-300 leading-relaxed">{currentTopicFocus}</p>
              </div>
            )}
            {unresolvedIssues.length > 0 && (
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-100 dark:border-slate-700 px-4 py-3">
                <p className="text-[10px] font-semibold text-muted dark:text-slate-500 uppercase tracking-wider mb-1.5">
                  Open questions
                </p>
                <ul className="space-y-1">
                  {unresolvedIssues.slice(0, 3).map((issue, i) => (
                    <li key={i} className="text-xs text-ink dark:text-slate-300 leading-relaxed">· {issue}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
