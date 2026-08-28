"use client";

import { useState } from "react";
import { StructuredAssessmentResult } from "@/types/assessment";
import { finalizeAssessment } from "@/lib/api";
import { FileCheck, X, AlertTriangle, ShieldCheck, Lock } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  structuredResult: StructuredAssessmentResult;
  onFinalized: (updatedResult: StructuredAssessmentResult) => void;
}

export default function FinalizeAssessmentModal({
  isOpen,
  onClose,
  structuredResult,
  onFinalized,
}: Props) {
  const [reviewer, setReviewer] = useState("Dr. Smith");
  const [reason, setReason] = useState("Teacher finalized assessment results");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const isAlreadyFinalized = structuredResult.assessment_status === "FINALIZED";

  const handleConfirmFinalize = async () => {
    setIsSubmitting(true);
    setError(null);
    try {
      const updated = await finalizeAssessment(structuredResult.assessment_id, {
        reviewer: reviewer.trim() || "Teacher",
        reason: reason.trim() || "Teacher finalized assessment results",
      });
      onFinalized(updated);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Finalization failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl max-w-lg w-full border border-slate-200 shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-emerald-50">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-xl bg-emerald-600 text-white flex items-center justify-center shadow-xs">
              <FileCheck size={20} />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                {isAlreadyFinalized ? "Finalized Assessment Review" : "Finalize Assessment"}
              </h2>
              <p className="text-xs text-slate-500">
                Revision Index: #{structuredResult.revision_index}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-slate-200 text-slate-500">
            <X size={18} />
          </button>
        </div>

        {/* Body Summary */}
        <div className="p-5 space-y-4">
          <div className="bg-slate-50 p-4 rounded-2xl border border-slate-200/80 space-y-2">
            <div className="text-xs font-bold text-slate-700">Assessment Summary</div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-slate-400">Total Questions:</span>{" "}
                <span className="font-semibold text-slate-900">{structuredResult.total_questions}</span>
              </div>
              <div>
                <span className="text-slate-400">Answered:</span>{" "}
                <span className="font-semibold text-slate-900">{structuredResult.answered_questions}</span>
              </div>
              <div>
                <span className="text-slate-400">Unanswered:</span>{" "}
                <span className="font-semibold text-slate-900">{structuredResult.unanswered_questions}</span>
              </div>
              <div>
                <span className="text-slate-400">Review Required:</span>{" "}
                <span className="font-semibold text-amber-600">{structuredResult.questions_needing_review}</span>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-200 flex items-baseline justify-between">
              <span className="text-xs font-semibold text-slate-600">Final Score:</span>
              <div className="text-right">
                <span className="text-xl font-extrabold text-slate-900">
                  {structuredResult.final_awarded_marks} / {structuredResult.total_max_marks}
                </span>
                <span className="text-xs font-bold text-emerald-600 ml-2">
                  ({structuredResult.percentage}%)
                </span>
              </div>
            </div>
          </div>

          {!isAlreadyFinalized && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1">
                  Reviewer Name
                </label>
                <input
                  type="text"
                  value={reviewer}
                  onChange={(e) => setReviewer(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs text-slate-900 font-medium focus:outline-hidden focus:border-emerald-500"
                  placeholder="e.g. Dr. Smith"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1">
                  Finalization Note / Reason
                </label>
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  rows={2}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs text-slate-900 font-medium focus:outline-hidden focus:border-emerald-500"
                  placeholder="Reason for finalizing score..."
                />
              </div>
            </div>
          )}

          {error && (
            <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center gap-2">
              <AlertTriangle size={14} className="shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!isAlreadyFinalized ? (
            <div className="bg-amber-50 p-3 rounded-xl border border-amber-200 text-[11px] text-amber-800 space-y-1">
              <div className="font-bold flex items-center gap-1">
                <Lock size={12} /> Finalization Impact:
              </div>
              <p>
                Status will change to <b>FINALIZED</b>. An immutable JSON snapshot will be saved with a SHA-256 integrity hash. Subsequent teacher overrides will create a new version revision.
              </p>
            </div>
          ) : (
            <div className="bg-emerald-50 p-3 rounded-xl border border-emerald-200 text-[11px] text-emerald-800 flex items-center gap-2 font-semibold">
              <ShieldCheck size={16} />
              <span>This assessment revision is currently finalized and locked.</span>
            </div>
          )}
        </div>

        {/* Footer Buttons */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-200 transition-colors"
          >
            Close
          </button>

          {!isAlreadyFinalized && (
            <button
              onClick={handleConfirmFinalize}
              disabled={isSubmitting}
              className="px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold transition-all shadow-sm flex items-center gap-1.5"
            >
              <FileCheck size={14} />
              {isSubmitting ? "Finalizing..." : "Confirm & Finalize Assessment"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
