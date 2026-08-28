"use client";

import { StudentAssessmentReport } from "@/types/assessment";
import { Lightbulb, Target } from "lucide-react";

interface Props {
  report: StudentAssessmentReport;
}

export default function ImprovementRecommendations({ report }: Props) {
  if (report.recommendations.length === 0) return null;

  return (
    <div className="bg-amber-50/70 border border-amber-200/90 rounded-3xl p-5 space-y-3">
      <div className="flex items-center gap-2 text-amber-900">
        <div className="h-8 w-8 rounded-xl bg-amber-500 text-white flex items-center justify-center shadow-xs">
          <Lightbulb size={18} />
        </div>
        <div>
          <h3 className="text-sm font-black">Targeted Improvement Recommendations</h3>
          <p className="text-[11px] text-amber-700 font-medium">Actionable study guidance</p>
        </div>
      </div>

      <div className="space-y-2 text-xs">
        {report.recommendations.map((rec, idx) => (
          <div
            key={idx}
            className="p-3 bg-white/90 border border-amber-200 rounded-2xl text-slate-800 font-medium flex items-start gap-2.5 shadow-2xs"
          >
            <Target size={14} className="text-amber-600 shrink-0 mt-0.5" />
            <span>{rec}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
