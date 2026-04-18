import { useState } from "react";
import { Check, Lock, Pencil, Plus, Trash2, X } from "lucide-react";
import { ActionItem } from "../types/meeting";

interface ActionItemsPanelProps {
  actionItems: ActionItem[];
  onAdd: (payload: {
    task: string;
    owner: string;
    deadline: string;
    priority: string;
    evidence: string;
  }) => void;
  onUpdate: (itemId: string, updates: Partial<ActionItem>) => void;
  onDelete: (itemId: string) => void;
}

function priorityStyle(priority: string) {
  if (priority === "high")
    return "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400";
  if (priority === "medium")
    return "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400";
  return "bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400";
}

const inputCls =
  "w-full border border-slate-200 dark:border-slate-600 rounded-lg px-3 py-2 text-sm text-ink dark:text-slate-100 bg-white dark:bg-slate-800 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition";

const selectCls =
  "border border-slate-200 dark:border-slate-600 rounded-lg px-3 py-2 text-sm text-ink dark:text-slate-100 bg-white dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition";

function ActionCard({
  item,
  onSave,
  onDelete,
}: {
  item: ActionItem;
  onSave: (updates: Partial<ActionItem>) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({
    task: item.task, owner: item.owner, deadline: item.deadline,
    priority: item.priority, status: item.status, evidence: item.evidence,
  });

  const save = () => { onSave(draft); setEditing(false); };
  const cancel = () => {
    setDraft({ task: item.task, owner: item.owner, deadline: item.deadline, priority: item.priority, status: item.status, evidence: item.evidence });
    setEditing(false);
  };

  return (
    <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 last:border-0">
      {editing ? (
        <div className="space-y-2.5">
          <input autoFocus className={inputCls} value={draft.task}
            onChange={(e) => setDraft((d) => ({ ...d, task: e.target.value }))}
            placeholder="Task description" />
          <div className="grid grid-cols-2 gap-2">
            <input className={inputCls} value={draft.owner}
              onChange={(e) => setDraft((d) => ({ ...d, owner: e.target.value }))}
              placeholder="Owner" />
            <input className={inputCls} value={draft.deadline}
              onChange={(e) => setDraft((d) => ({ ...d, deadline: e.target.value }))}
              placeholder="Deadline" />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <select className={selectCls} value={draft.priority}
              onChange={(e) => setDraft((d) => ({ ...d, priority: e.target.value }))}>
              <option value="high">High priority</option>
              <option value="medium">Medium priority</option>
              <option value="low">Low priority</option>
            </select>
            <select className={selectCls} value={draft.status}
              onChange={(e) => setDraft((d) => ({ ...d, status: e.target.value }))}>
              <option value="pending">Pending</option>
              <option value="completed">Completed</option>
            </select>
          </div>
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
            <p className="flex-1 text-sm font-medium text-ink dark:text-slate-100 leading-relaxed">{item.task}</p>
            <div className="flex items-center gap-0.5 shrink-0 ml-1">
              {item.human_locked && <span title="Human locked" className="p-1 text-amber-500"><Lock size={11} /></span>}
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
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${priorityStyle(item.priority)}`}>
              {item.priority}
            </span>
            {item.needs_review && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-400">
                Review needed
              </span>
            )}
            {item.status === "completed" && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400">
                Done
              </span>
            )}
            {item.owner && item.owner !== "unknown" && (
              <span className="text-[11px] text-muted dark:text-slate-400">{item.owner}</span>
            )}
            {item.deadline && item.deadline !== "unspecified" && (
              <span className="text-[11px] text-muted dark:text-slate-400">· {item.deadline}</span>
            )}
          </div>
          {item.evidence && (
            <p className="mt-1.5 text-xs text-muted dark:text-slate-400 leading-relaxed line-clamp-2">
              {item.evidence}
            </p>
          )}
        </>
      )}
    </div>
  );
}

export function ActionItemsPanel({ actionItems, onAdd, onUpdate, onDelete }: ActionItemsPanelProps) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState({ task: "", owner: "", deadline: "", priority: "medium", evidence: "" });

  const submitAdd = () => {
    if (!draft.task.trim()) return;
    onAdd({ ...draft, deadline: draft.deadline || "unspecified" });
    setDraft({ task: "", owner: "", deadline: "", priority: "medium", evidence: "" });
    setAdding(false);
  };

  return (
    <section className="bg-surface dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden shadow-card">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-ink dark:text-slate-100">Action Items</h3>
          {actionItems.length > 0 && (
            <p className="text-[11px] text-muted dark:text-slate-500 mt-0.5">
              {actionItems.length} item{actionItems.length !== 1 ? "s" : ""}
            </p>
          )}
        </div>
        <button type="button" onClick={() => setAdding((v) => !v)}
          className="flex items-center gap-1 text-xs text-accent hover:text-blue-700 dark:hover:text-blue-400 font-medium transition">
          <Plus size={13} /> {adding ? "Cancel" : "Add"}
        </button>
      </div>

      {adding && (
        <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 space-y-2.5">
          <input autoFocus className={inputCls} value={draft.task}
            onChange={(e) => setDraft((d) => ({ ...d, task: e.target.value }))}
            placeholder="What needs to happen?"
            onKeyDown={(e) => { if (e.key === "Enter") submitAdd(); }} />
          <div className="grid grid-cols-2 gap-2">
            <input className={inputCls} value={draft.owner}
              onChange={(e) => setDraft((d) => ({ ...d, owner: e.target.value }))}
              placeholder="Owner" />
            <input className={inputCls} value={draft.deadline}
              onChange={(e) => setDraft((d) => ({ ...d, deadline: e.target.value }))}
              placeholder="Deadline (optional)" />
          </div>
          <div className="grid grid-cols-[160px_1fr] gap-2">
            <select className={selectCls} value={draft.priority}
              onChange={(e) => setDraft((d) => ({ ...d, priority: e.target.value }))}>
              <option value="high">High priority</option>
              <option value="medium">Medium priority</option>
              <option value="low">Low priority</option>
            </select>
            <input className={inputCls} value={draft.evidence}
              onChange={(e) => setDraft((d) => ({ ...d, evidence: e.target.value }))}
              placeholder="Evidence (optional)" />
          </div>
          <button type="button" onClick={submitAdd} disabled={!draft.task.trim()}
            className="px-3 py-1.5 bg-ink dark:bg-slate-100 text-white dark:text-slate-900 rounded-lg text-xs font-medium hover:bg-slate-700 dark:hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition">
            Add action item
          </button>
        </div>
      )}

      <div>
        {actionItems.length === 0 && !adding ? (
          <p className="px-5 py-4 text-sm text-muted dark:text-slate-500">
            Action items will appear here as LiveNote detects clear commitments.
          </p>
        ) : (
          actionItems.map((item) => (
            <ActionCard key={item.id} item={item}
              onSave={(updates) => onUpdate(item.id, updates)}
              onDelete={() => onDelete(item.id)} />
          ))
        )}
      </div>
    </section>
  );
}
