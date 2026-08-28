"use client";

import { StudentAssessmentReport } from "@/types/assessment";
import { CheckCircle2, AlertTriangle, XCircle, HelpCircle } from "lucide-react";

interface Props {
  report: StudentAssessmentReport;
}

export default function PerformanceBreakdown({ report }: Props) {
  const allCriteria = report.question_results.flatMap((q) => q.criteria_summary);
  const totalCount = allCriteria.length;

  const presentCount = allCriteria.filter((c) => c.evidence_status === "present").length;
  const partialCount = allCriteria.filter((c) => c.evidence_status === "partially_present").length;
  const missingCount = allCriteria.filter((c) => c.evidence_status === "missing" || c.evidence_status === "contradicted").length;

  return (
    <div className="bg-slate-50 border border-slate-200/80 rounded-3xl p-5 space-y-4">
      <h3 className="text-sm font-black text-slate-900">
        Rubric Mastery & Criterion Breakdown ({totalCount} Total Criteria)
      </h3>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-emerald-700 flex items-center gap-1">
              <CheckCircle2 size={14} /> Fully Demonstrated ({presentCount})
            </span>
            <span className="font-mono font-bold text-slate-700">
              {totalCount > 0 ? Math.round((presentCount / totalCount) * 100) : 0}%
            </span>
          </div>
          <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
            <div
              className="bg-emerald-500 h-full rounded-full transition-all"
              style={{ width: `${totalCount > 0 ? (presentCount / totalCount) * 100 : 0}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-amber-700 flex items-center gap-1">
              <AlertTriangle size={14} /> Partially Demonstrated ({partialCount})
            </span>
            <span className="font-mono font-bold text-slate-700">
              {totalCount > 0 ? Math.round((partialCount / totalCount) * 100) : 0}%
            </span>
          </div>
          <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
            <div
              className="bg-amber-500 h-full rounded-full transition-all"
              style={{ width: `${totalCount > 0 ? (partialCount / totalCount) * 100 : 0}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-rose-700 flex items-center gap-1">
              <XCircle size={14} /> Missing / Contradicted ({missingCount})
            </span>
            <span className="font-mono font-bold text-slate-700">
              {totalCount > 0 ? Math.round((missingCount / totalCount) * 100) : 0}%
            </span>
          </div>
          <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
            <div
              className="bg-rose-500 h-full rounded-full transition-all"
              style={{ width: `${totalCount > 0 ? (missingCount / totalCount) * 100 : 0}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
