"use client";

import { StructuredQuestionResult } from "@/types/assessment";
import { HelpCircle, AlertTriangle, X, ArrowRight, ShieldAlert } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  reviewQuestions: StructuredQuestionResult[];
  onSelectQuestion: (questionId: string) => void;
}

export default function TeacherReviewQueueModal({
  isOpen,
  onClose,
  reviewQuestions,
  onSelectQuestion,
}: Props) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl max-w-2xl w-full border border-slate-200 shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-amber-50/50">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-xl bg-amber-500 text-white flex items-center justify-center shadow-xs">
              <HelpCircle size={18} />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900">Teacher Review Queue</h2>
              <p className="text-xs text-slate-500">
                {reviewQuestions.length} question{reviewQuestions.length === 1 ? "" : "s"} flagged for teacher inspection
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* List Body */}
        <div className="p-4 overflow-y-auto space-y-3 flex-1">
          {reviewQuestions.length === 0 ? (
            <div className="p-8 text-center bg-slate-50 rounded-2xl border border-slate-200/80">
              <ShieldAlert className="mx-auto text-emerald-500 mb-2" size={32} />
              <h3 className="text-sm font-bold text-slate-800">Review Queue Empty!</h3>
              <p className="text-xs text-slate-500 mt-1">
                All questions have high confidence evaluation or have been reviewed.
              </p>
            </div>
          ) : (
            reviewQuestions.map((q) => (
              <div
                key={q.question_id}
                className="p-3.5 rounded-2xl bg-white border border-slate-200 hover:border-amber-400 shadow-2xs transition-all flex items-start justify-between gap-3"
              >
                <div className="space-y-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-extrabold text-xs text-slate-900 bg-slate-100 px-2 py-0.5 rounded-md">
                      Q{q.question_number}
                    </span>
                    <span className="text-[11px] font-bold text-amber-700 bg-amber-100/80 border border-amber-200 px-2 py-0.5 rounded-full">
                      Reason: {q.escalation_reason || "Low Confidence"}
                    </span>
                    <span className="text-[11px] font-mono font-semibold text-slate-500">
                      Conf: {Math.round(q.evaluation_confidence * 100)}%
                    </span>
                  </div>
                  <p className="text-xs text-slate-700 font-medium line-clamp-2">
                    {q.question_text}
                  </p>
                  <div className="text-[11px] text-slate-400">
                    Grading Method: <span className="font-mono text-slate-600">{q.grading_provenance || "local"}</span>
                  </div>
                </div>

                <button
                  onClick={() => {
                    onSelectQuestion(q.question_id);
                    onClose();
                  }}
                  className="px-3 py-1.5 rounded-xl bg-slate-900 text-white text-xs font-semibold hover:bg-amber-600 transition-colors flex items-center gap-1 shrink-0 self-center"
                >
                  Review <ArrowRight size={12} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
