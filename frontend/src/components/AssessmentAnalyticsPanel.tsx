"use client";

import { StructuredAssessmentResult } from "@/types/assessment";
import { BarChart3, X, CheckCircle2, AlertTriangle, ShieldCheck, PieChart } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  structuredResult: StructuredAssessmentResult;
}

export default function AssessmentAnalyticsPanel({
  isOpen,
  onClose,
  structuredResult,
}: Props) {
  if (!isOpen) return null;

  const confAvg = Math.round(structuredResult.overall_confidence * 100);
  const questions = structuredResult.question_results;

  const highConfCount = questions.filter((q) => q.evaluation_confidence >= 0.85).length;
  const medConfCount = questions.filter((q) => q.evaluation_confidence >= 0.6 && q.evaluation_confidence < 0.85).length;
  const lowConfCount = questions.filter((q) => q.evaluation_confidence < 0.6).length;

  const totalCriteria = questions.reduce((sum, q) => sum + q.criterion_results.length, 0);
  const presentCriteria = questions.reduce(
    (sum, q) => sum + q.criterion_results.filter((c) => c.evidence_status === "present").length,
    0
  );
  const partialCriteria = questions.reduce(
    (sum, q) => sum + q.criterion_results.filter((c) => c.evidence_status === "partially_present").length,
    0
  );
  const missingCriteria = questions.reduce(
    (sum, q) => sum + q.criterion_results.filter((c) => c.evidence_status === "missing" || c.evidence_status === "contradicted").length,
    0
  );

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-50 flex justify-end">
      <div className="bg-white max-w-xl w-full h-full shadow-2xl flex flex-col border-l border-slate-200">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <BarChart3 size={18} className="text-slate-900" />
            <h2 className="text-sm font-bold text-slate-900">Assessment Analytics & Insights</h2>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500">
            <X size={18} />
          </button>
        </div>

        {/* Analytics Body */}
        <div className="p-5 overflow-y-auto space-y-5 flex-1">
          {/* Card 1: Score & Confidence Summary */}
          <div className="bg-gradient-to-br from-slate-900 to-slate-800 text-white rounded-2xl p-4 space-y-2">
            <div className="text-xs font-semibold text-slate-300">Overall Performance & AI Confidence</div>
            <div className="flex items-baseline justify-between">
              <div>
                <span className="text-3xl font-extrabold">{structuredResult.final_awarded_marks}</span>
                <span className="text-sm text-slate-400 font-bold"> / {structuredResult.total_max_marks}</span>
              </div>
              <span className="text-lg font-black text-emerald-400">{structuredResult.percentage}%</span>
            </div>
            <div className="w-full bg-slate-700 h-2 rounded-full overflow-hidden mt-1">
              <div
                className="bg-emerald-400 h-full rounded-full transition-all"
                style={{ width: `${Math.min(100, structuredResult.percentage)}%` }}
              />
            </div>
            <div className="text-xs text-slate-400 flex items-center justify-between pt-1">
              <span>Overall Evaluation Confidence</span>
              <span className="font-mono font-bold text-slate-200">{confAvg}%</span>
            </div>
          </div>

          {/* Card 2: Confidence Distribution */}
          <div className="bg-slate-50 p-4 rounded-2xl border border-slate-200/80 space-y-2">
            <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
              <PieChart size={14} className="text-indigo-600" /> Evaluation Confidence Distribution
            </div>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-600">High Confidence (&ge; 85%):</span>
                <span className="font-bold text-emerald-700">{highConfCount} questions</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-600">Moderate Confidence (60% - 84%):</span>
                <span className="font-bold text-amber-700">{medConfCount} questions</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-600">Low Confidence (&lt; 60%):</span>
                <span className="font-bold text-rose-700">{lowConfCount} questions</span>
              </div>
            </div>
          </div>

          {/* Card 3: Criterion Mastery Breakdown */}
          <div className="bg-slate-50 p-4 rounded-2xl border border-slate-200/80 space-y-2">
            <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
              <CheckCircle2 size={14} className="text-emerald-600" /> Rubric Criterion Mastery (Total: {totalCriteria})
            </div>
            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between text-[11px] mb-0.5">
                  <span className="font-semibold text-emerald-700">Present (Fully Demonstrated):</span>
                  <span className="font-bold text-slate-900">{presentCriteria}</span>
                </div>
                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-emerald-500 h-full rounded-full"
                    style={{ width: `${totalCriteria > 0 ? (presentCriteria / totalCriteria) * 100 : 0}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-[11px] mb-0.5">
                  <span className="font-semibold text-amber-700">Partially Present:</span>
                  <span className="font-bold text-slate-900">{partialCriteria}</span>
                </div>
                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-amber-500 h-full rounded-full"
                    style={{ width: `${totalCriteria > 0 ? (partialCriteria / totalCriteria) * 100 : 0}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-[11px] mb-0.5">
                  <span className="font-semibold text-rose-700">Missing / Contradicted:</span>
                  <span className="font-bold text-slate-900">{missingCriteria}</span>
                </div>
                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-rose-500 h-full rounded-full"
                    style={{ width: `${totalCriteria > 0 ? (missingCriteria / totalCriteria) * 100 : 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
