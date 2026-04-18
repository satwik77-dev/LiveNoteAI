import { useState } from "react";
import { Check, Lock, Pencil, Plus, Trash2, X } from "lucide-react";
import { Decision } from "../types/meeting";

interface DecisionsPanelProps {
  decisions: Decision[];
  onAdd: (payload: { decision: string; evidence: string }) => void;
  onUpdate: (itemId: string, updates: Partial<Decision>) => void;
  onDelete: (itemId: string) => void;
}

const inputCls =
  "w-full border border-slate-200 dark:border-slate-600 rounded-lg px-3 py-2 text-sm text-ink dark:text-slate-100 bg-white dark:bg-slate-800 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition";

function DecisionCard({
  decision,
  onSave,
  onDelete,
}: {
  decision: Decision;
  onSave: (updates: Partial<Decision>) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ decision: decision.decision, evidence: decision.evidence });

  const save = () => { onSave(draft); setEditing(false); };
  const cancel = () => { setDraft({ decision: decision.decision, evidence: decision.evidence }); setEditing(false); };

  return (
    <div className="px-4 py-3.5 border-b border-slate-100 dark:border-slate-800 last:border-0">
      {editing ? (
        <div className="space-y-2">
          <textarea autoFocus className={`${inputCls} resize-none`} rows={2} value={draft.decision}
            onChange={(e) => setDraft((d) => ({ ...d, decision: e.target.value }))}
            placeholder="Decision" />
          <textarea className={`${inputCls} resize-none`} rows={2} value={draft.evidence}
            onChange={(e) => setDraft((d) => ({ ...d, evidence: e.target.value }))}
            placeholder="Evidence (optional)" />
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
        <>
          <div className="flex items-start gap-2">
            <p className="flex-1 text-sm text-ink dark:text-slate-200 leading-relaxed">{decision.decision}</p>
            <div className="flex items-center gap-0.5 shrink-0">
              {decision.human_locked && <span title="Human locked" className="p-1 text-amber-500"><Lock size={11} /></span>}
              <button type="button" onClick={() => setEditing(true)}
                className="p-1 rounded text-muted dark:text-slate-500 hover:text-ink dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition">
                <Pencil size={12} />
              </button>
              <button type="button" onClick={onDelete}
                className="p-1 rounded text-muted dark:text-slate-500 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 transition">
                <Trash2 size={12} />
              </button>
            </div>
          </div>
          {decision.evidence && (
            <p className="mt-1 text-[11px] text-muted dark:text-slate-400 leading-relaxed line-clamp-2">
              {decision.evidence}
            </p>
          )}
        </>
      )}
    </div>
  );
}

export function DecisionsPanel({ decisions, onAdd, onUpdate, onDelete }: DecisionsPanelProps) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState({ decision: "", evidence: "" });

  const submitAdd = () => {
    if (!draft.decision.trim()) return;
    onAdd(draft);
    setDraft({ decision: "", evidence: "" });
    setAdding(false);
  };

  return (
    <section className="bg-surface dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden shadow-card">
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-ink dark:text-slate-100">Decisions</h3>
          {decisions.length > 0 && (
            <p className="text-[11px] text-muted dark:text-slate-500 mt-0.5">{decisions.length} recorded</p>
          )}
        </div>
        <button type="button" onClick={() => setAdding((v) => !v)}
          className="flex items-center gap-1 text-xs text-accent hover:text-blue-700 dark:hover:text-blue-400 font-medium transition">
          <Plus size={13} /> {adding ? "Cancel" : "Add"}
        </button>
      </div>

      {adding && (
        <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 space-y-2">
          <textarea autoFocus className={`${inputCls} resize-none`} rows={2} value={draft.decision}
            onChange={(e) => setDraft((d) => ({ ...d, decision: e.target.value }))}
            placeholder="What was decided?" />
          <input className={inputCls} value={draft.evidence}
            onChange={(e) => setDraft((d) => ({ ...d, evidence: e.target.value }))}
            placeholder="Evidence (optional)" />
          <button type="button" onClick={submitAdd} disabled={!draft.decision.trim()}
            className="px-3 py-1.5 bg-ink dark:bg-slate-100 text-white dark:text-slate-900 rounded-lg text-xs font-medium hover:bg-slate-700 dark:hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition">
            Add decision
          </button>
        </div>
      )}

      <div>
        {decisions.length === 0 && !adding ? (
          <p className="px-4 py-4 text-sm text-muted dark:text-slate-500">
            Decisions will appear once the conversation reaches a clear resolution.
          </p>
        ) : (
          decisions.map((d) => (
            <DecisionCard key={d.id} decision={d}
              onSave={(updates) => onUpdate(d.id, updates)}
              onDelete={() => onDelete(d.id)} />
          ))
        )}
      </div>
    </section>
  );
}
