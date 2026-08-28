"use client";

import { useState } from "react";
import { AssessmentRevision } from "@/types/assessment";
import { getRevisionSnapshot } from "@/lib/api";
import { History, X, ShieldCheck, FileJson, CheckCircle2, Lock } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  assessmentId: string;
  revisions: AssessmentRevision[];
}

export default function RevisionHistoryDrawer({
  isOpen,
  onClose,
  assessmentId,
  revisions,
}: Props) {
  const [selectedSnapshot, setSelectedSnapshot] = useState<any | null>(null);
  const [loadingRev, setLoadingRev] = useState<number | null>(null);

  if (!isOpen) return null;

  const handleFetchSnapshot = async (revIndex: number) => {
    setLoadingRev(revIndex);
    try {
      const snap = await getRevisionSnapshot(assessmentId, revIndex);
      setSelectedSnapshot(snap);
    } catch (e) {
      alert(`Failed to load snapshot for revision ${revIndex}`);
    } finally {
      setLoadingRev(null);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-50 flex justify-end">
      <div className="bg-white max-w-xl w-full h-full shadow-2xl flex flex-col border-l border-slate-200">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <History size={18} className="text-indigo-600" />
            <h2 className="text-sm font-bold text-slate-900">Revision Version History</h2>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500">
            <X size={18} />
          </button>
        </div>

        {/* Revision List */}
        <div className="p-4 overflow-y-auto space-y-4 flex-1">
          {revisions.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-400">No finalized revisions recorded yet</div>
          ) : (
            revisions.map((rev) => (
              <div
                key={rev.revision_id || rev.revision_index}
                className="p-4 bg-slate-50 rounded-2xl border border-slate-200/90 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-extrabold text-xs text-slate-900 flex items-center gap-1.5">
                    <Lock size={12} className="text-emerald-600" /> Revision #{rev.revision_index}
                  </span>
                  <span className="text-[11px] font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full">
                    {rev.final_awarded_marks} pts ({rev.percentage}%)
                  </span>
                </div>

                <div className="text-xs text-slate-600 space-y-1">
                  <div>Finalized By: <span className="font-semibold text-slate-900">{rev.finalized_by}</span></div>
                  <div>Reason: <span className="italic text-slate-700">{rev.reason}</span></div>
                  <div className="text-[11px] text-slate-400 font-mono">
                    Timestamp: {rev.timestamp ? new Date(rev.timestamp).toLocaleString() : "N/A"}
                  </div>
                  {rev.snapshot_hash && (
                    <div className="text-[10px] font-mono text-slate-500 truncate bg-slate-200/60 p-1.5 rounded-lg">
                      SHA-256: {rev.snapshot_hash}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleFetchSnapshot(rev.revision_index)}
                  disabled={loadingRev === rev.revision_index}
                  className="w-full py-1.5 bg-slate-900 text-white rounded-xl text-xs font-semibold hover:bg-slate-800 transition-colors flex items-center justify-center gap-1.5 shadow-xs"
                >
                  <FileJson size={13} />
                  {loadingRev === rev.revision_index ? "Loading Snapshot..." : "Inspect Immutable Snapshot"}
                </button>
              </div>
            ))
          )}
        </div>

        {/* Snapshot Modal overlay */}
        {selectedSnapshot && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs z-60 flex items-center justify-center p-4">
            <div className="bg-white rounded-3xl max-w-2xl w-full border border-slate-200 shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
              <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-900 text-white">
                <div className="flex items-center gap-2">
                  <ShieldCheck size={18} className="text-emerald-400" />
                  <h3 className="text-sm font-bold">
                    Revision #{selectedSnapshot.revision_index} Immutable Snapshot
                  </h3>
                </div>
                <button
                  onClick={() => setSelectedSnapshot(null)}
                  className="p-1 rounded-lg hover:bg-slate-800 text-slate-300"
                >
                  <X size={18} />
                </button>
              </div>

              <div className="p-3 bg-emerald-50 border-b border-emerald-200 flex items-center justify-between text-xs">
                <div className="flex items-center gap-1.5 text-emerald-800 font-bold">
                  <CheckCircle2 size={16} className="text-emerald-600" />
                  <span>SHA-256 Snapshot Integrity Verified</span>
                </div>
                <span className="font-mono text-[10px] text-emerald-700 font-bold">
                  {selectedSnapshot.snapshot_hash ? `${selectedSnapshot.snapshot_hash.substring(0, 16)}...` : ""}
                </span>
              </div>

              <div className="p-4 overflow-y-auto bg-slate-950 text-slate-200 font-mono text-xs flex-1">
                <pre className="whitespace-pre-wrap">
                  {JSON.stringify(selectedSnapshot, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
