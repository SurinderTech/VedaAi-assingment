"use client";

import { X, Sparkles } from "lucide-react";
import AssessmentInsightsPanel from "./AssessmentInsightsPanel";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  assessmentId: string;
  onSelectQuestion: (qId: string) => void;
}

export default function InsightsDrawer({
  isOpen,
  onClose,
  assessmentId,
  onSelectQuestion,
}: Props) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-50 flex justify-end">
      <div className="bg-white max-w-3xl w-full h-full shadow-2xl flex flex-col border-l border-slate-200 overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50 shrink-0">
          <div className="flex items-center gap-2">
            <Sparkles size={18} className="text-amber-500" />
            <h2 className="text-sm font-bold text-slate-900">Step 9 Assessment Intelligence & Teacher Insights</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <AssessmentInsightsPanel
            assessmentId={assessmentId}
            onSelectQuestion={(qId) => {
              onSelectQuestion(qId);
              onClose();
            }}
          />
        </div>
      </div>
    </div>
  );
}
