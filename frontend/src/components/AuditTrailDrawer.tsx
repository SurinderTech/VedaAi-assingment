"use client";

import { AuditEvent } from "@/types/assessment";
import { ShieldCheck, X, Clock, User, FileText } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  auditEvents: AuditEvent[];
}

export default function AuditTrailDrawer({ isOpen, onClose, auditEvents }: Props) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-50 flex justify-end">
      <div className="bg-white max-w-lg w-full h-full shadow-2xl flex flex-col border-l border-slate-200">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <ShieldCheck size={18} className="text-emerald-600" />
            <h2 className="text-sm font-bold text-slate-900">Immutable Audit Trail</h2>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500">
            <X size={18} />
          </button>
        </div>

        {/* Timeline */}
        <div className="p-4 overflow-y-auto space-y-4 flex-1">
          {auditEvents.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-400">No audit events recorded</div>
          ) : (
            auditEvents.map((evt, idx) => (
              <div key={evt.event_id || idx} className="relative pl-6 pb-2 border-l-2 border-slate-200 last:border-l-0">
                <div className="absolute -left-[9px] top-0.5 h-4 w-4 rounded-full bg-slate-900 border-2 border-white" />
                <div className="bg-slate-50 p-3 rounded-2xl border border-slate-200/80 space-y-1.5">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-bold text-slate-900 px-2 py-0.5 bg-slate-200 rounded-md">
                      {evt.event_type}
                    </span>
                    <span className="text-slate-400 flex items-center gap-1 font-mono">
                      <Clock size={10} /> {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : "Now"}
                    </span>
                  </div>

                  {evt.question_id && (
                    <div className="text-xs font-semibold text-slate-800">
                      Question: <span className="font-mono text-indigo-600">Q{evt.question_id}</span>
                    </div>
                  )}

                  <div className="text-xs text-slate-600 flex items-center gap-1">
                    <User size={12} className="text-slate-400" />
                    Source: <span className="font-medium text-slate-900">{evt.source}</span>
                  </div>

                  {evt.reason && (
                    <div className="text-xs text-slate-500 bg-white p-2 rounded-xl border border-slate-200/60">
                      Reason: <span className="italic">{evt.reason}</span>
                    </div>
                  )}

                  {(evt.previous_value || evt.new_value) && (
                    <div className="text-[11px] font-mono bg-slate-900 text-slate-200 p-2 rounded-xl overflow-x-auto space-y-1">
                      {evt.previous_value && <div>Prev: {JSON.stringify(evt.previous_value)}</div>}
                      {evt.new_value && <div>New:  {JSON.stringify(evt.new_value)}</div>}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
