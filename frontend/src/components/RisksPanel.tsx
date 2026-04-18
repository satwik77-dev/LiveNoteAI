import { useState } from "react";
import { Check, Lock, Pencil, Plus, Trash2, X } from "lucide-react";
import { Risk } from "../types/meeting";

interface RisksPanelProps {
  risks: Risk[];
  onAdd: (payload: { risk: string; evidence: string }) => void;
  onUpdate: (itemId: string, updates: Partial<Risk>) => void;
  onDelete: (itemId: string) => void;
}

const inputCls =
  "w-full border border-slate-200 dark:border-slate-600 rounded-lg px-3 py-2 text-sm text-ink dark:text-slate-100 bg-white dark:bg-slate-800 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition";

function RiskCard({
  risk,
  onSave,
  onDelete,
}: {
  risk: Risk;
  onSave: (updates: Partial<Risk>) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ risk: risk.risk, evidence: risk.evidence });

  const save = () => { onSave(draft); setEditing(false); };
  const cancel = () => { setDraft({ risk: risk.risk, evidence: risk.evidence }); setEditing(false); };

  return (
    <div className="px-4 py-3.5 border-b border-slate-100 dark:border-slate-800 last:border-0">
      {editing ? (
        <div className="space-y-2">
          <textarea autoFocus className={`${inputCls} resize-none`} rows={2} value={draft.risk}
            onChange={(e) => setDraft((d) => ({ ...d, risk: e.target.value }))}
            placeholder="Risk or blocker" />
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
            <p className="flex-1 text-sm text-ink dark:text-slate-200 leading-relaxed">{risk.risk}</p>
            <div className="flex items-center gap-0.5 shrink-0">
              {risk.human_locked && <span title="Human locked" className="p-1 text-amber-500"><Lock size={11} /></span>}
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
          {risk.evidence && (
            <p className="mt-1 text-[11px] text-muted dark:text-slate-400 leading-relaxed line-clamp-2">
              {risk.evidence}
            </p>
          )}
        </>
      )}
    </div>
  );
}

export function RisksPanel({ risks, onAdd, onUpdate, onDelete }: RisksPanelProps) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState({ risk: "", evidence: "" });

  const submitAdd = () => {
    if (!draft.risk.trim()) return;
    onAdd(draft);
    setDraft({ risk: "", evidence: "" });
    setAdding(false);
  };

  return (
    <section className="bg-surface dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden shadow-card">
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-ink dark:text-slate-100">Risks</h3>
          {risks.length > 0 && (
            <p className="text-[11px] text-muted dark:text-slate-500 mt-0.5">{risks.length} flagged</p>
          )}
        </div>
        <button type="button" onClick={() => setAdding((v) => !v)}
          className="flex items-center gap-1 text-xs text-accent hover:text-blue-700 dark:hover:text-blue-400 font-medium transition">
          <Plus size={13} /> {adding ? "Cancel" : "Add"}
        </button>
      </div>

      {adding && (
        <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 space-y-2">
          <textarea autoFocus className={`${inputCls} resize-none`} rows={2} value={draft.risk}
            onChange={(e) => setDraft((d) => ({ ...d, risk: e.target.value }))}
            placeholder="What could block progress?" />
          <input className={inputCls} value={draft.evidence}
            onChange={(e) => setDraft((d) => ({ ...d, evidence: e.target.value }))}
            placeholder="Evidence (optional)" />
          <button type="button" onClick={submitAdd} disabled={!draft.risk.trim()}
            className="px-3 py-1.5 bg-ink dark:bg-slate-100 text-white dark:text-slate-900 rounded-lg text-xs font-medium hover:bg-slate-700 dark:hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition">
            Add risk
          </button>
        </div>
      )}

      <div>
        {risks.length === 0 && !adding ? (
          <p className="px-4 py-4 text-sm text-muted dark:text-slate-500">
            Risks and blockers will appear here after LiveNote validates them.
          </p>
        ) : (
          risks.map((r) => (
            <RiskCard key={r.id} risk={r}
              onSave={(updates) => onUpdate(r.id, updates)}
              onDelete={() => onDelete(r.id)} />
          ))
        )}
      </div>
    </section>
  );
}
