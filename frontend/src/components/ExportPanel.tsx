import { FileDown } from "lucide-react";
import { ExportBundle } from "../types/meeting";

interface ExportPanelProps {
  exportBundle: ExportBundle | null;
  sessionActive: boolean;
}

function downloadBase64(filename: string, mimeType: string, base64: string) {
  const bytes = Uint8Array.from(window.atob(base64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function ExportPanel({ exportBundle }: ExportPanelProps) {
  if (!exportBundle) return null;

  const title = exportBundle.report.meeting.title || "meeting-report";

  return (
    <section className="bg-surface dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden shadow-card">
      <div className="px-5 py-3.5 border-b border-slate-100 dark:border-slate-800">
        <h3 className="text-sm font-semibold text-ink dark:text-slate-100">Export</h3>
        <p className="text-xs text-muted dark:text-slate-500 mt-0.5">
          Your meeting report is ready to download.
        </p>
      </div>
      <div className="px-5 py-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => downloadBase64(`${title}.pdf`, "application/pdf", exportBundle.pdfBase64)}
          className="flex items-center gap-2 px-4 py-2 bg-ink dark:bg-slate-100 text-white dark:text-slate-900 rounded-lg text-sm font-medium hover:bg-slate-700 dark:hover:bg-white transition"
        >
          <FileDown size={14} />
          Export as PDF
        </button>
        <button
          type="button"
          onClick={() => downloadBase64(`${title}.json`, "application/json", exportBundle.jsonBase64)}
          className="flex items-center gap-2 px-4 py-2 border border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-lg text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition"
        >
          Download JSON
        </button>
      </div>
    </section>
  );
}
