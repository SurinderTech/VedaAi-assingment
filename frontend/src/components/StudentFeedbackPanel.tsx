"use client";

import { StudentAssessmentReport } from "@/types/assessment";
import { Sparkles, CheckCircle2, FileText } from "lucide-react";

interface Props {
  report: StudentAssessmentReport;
}

export default function StudentFeedbackPanel({ report }: Props) {
  return (
    <div className="bg-white border border-slate-200 rounded-3xl p-5 space-y-4 shadow-xs">
      <div className="flex items-center gap-2">
        <div className="h-8 w-8 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
          <Sparkles size={18} />
        </div>
        <div>
          <h3 className="text-sm font-black text-slate-900">Overall Assessment Feedback</h3>
          <p className="text-[11px] text-slate-500 font-medium">Derived strictly from evidence</p>
        </div>
      </div>

      <p className="text-xs text-slate-700 leading-relaxed font-medium bg-slate-50 p-4 rounded-2xl border border-slate-200/80">
        {report.feedback}
      </p>

      {report.strengths.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
            <CheckCircle2 size={14} className="text-emerald-600" /> Key Demonstrated Strengths
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            {report.strengths.map((str, idx) => (
              <div key={idx} className="p-3 bg-emerald-50/60 border border-emerald-200/80 rounded-2xl text-emerald-900 font-medium">
                • {str}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
