"use client";

import { StudentPerformanceSummary } from "@/types/assessment";
import { exportStudentReportUrl } from "@/lib/api";
import { Award, Download, CheckCircle2, Circle, AlertCircle, ShieldCheck } from "lucide-react";

interface Props {
  summary: StudentPerformanceSummary;
  revisionIndex?: number;
}

export default function StudentResultOverview({ summary, revisionIndex }: Props) {
  const exportUrl = exportStudentReportUrl(summary.assessment_id, revisionIndex);

  let bandColor = "bg-emerald-100 text-emerald-800 border-emerald-300";
  if (summary.performance_band === "Developing") {
    bandColor = "bg-amber-100 text-amber-800 border-amber-300";
  } else if (summary.performance_band === "Needs Support") {
    bandColor = "bg-rose-100 text-rose-800 border-rose-300";
  }

  return (
    <div className="bg-white border-b border-slate-200 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-2xl bg-slate-900 text-white flex items-center justify-center font-black text-base shadow-sm">
              vAI
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-black text-slate-900 tracking-tight">
                  Student Assessment Performance
                </h1>
                <span className={`px-3 py-0.5 rounded-full text-xs font-extrabold border ${bandColor}`}>
                  {summary.performance_band}
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium">
                Assessment ID: <span className="font-mono text-slate-700">{summary.assessment_id}</span>
              </p>
            </div>
          </div>

          <a
            href={exportUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold transition-all shadow-xs flex items-center gap-1.5 shrink-0 self-start sm:self-center"
          >
            <Download size={14} />
            Export Printable Report
          </a>
        </div>

        {/* Metric Cards Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-gradient-to-br from-slate-900 to-slate-800 text-white rounded-2xl p-4">
            <div className="text-[11px] font-semibold text-slate-300">Final Score</div>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-black">{summary.final_awarded_marks}</span>
              <span className="text-xs text-slate-400 font-bold">/ {summary.total_max_marks}</span>
            </div>
            <div className="mt-1 text-xs font-bold text-emerald-400">{summary.percentage}%</div>
          </div>

          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4">
            <div className="text-[11px] font-semibold text-slate-500">Answered Questions</div>
            <div className="text-xl font-bold text-slate-800 mt-1">
              {summary.answered_questions} <span className="text-xs font-normal text-slate-500">submitted</span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              Unanswered: {summary.unanswered_questions}
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4">
            <div className="text-[11px] font-semibold text-slate-500">Evaluation Confidence</div>
            <div className="text-xl font-bold text-emerald-600 mt-1">
              {Math.round(summary.overall_confidence * 100)}%
            </div>
            <div className="text-[10px] text-slate-400 mt-1">Evidence-Grounded</div>
          </div>

          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4">
            <div className="text-[11px] font-semibold text-slate-500">Performance Band</div>
            <div className="text-xl font-bold text-indigo-600 mt-1">
              {summary.performance_band}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">Display Label Only</div>
          </div>
        </div>
      </div>
    </div>
  );
}
