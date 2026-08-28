"use client";

import { StructuredAssessmentResult } from "@/types/assessment";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Circle,
  FileCheck,
  History,
  ShieldCheck,
  BarChart3,
  Lock,
  ArrowUpRight,
} from "lucide-react";

interface Props {
  structuredResult: StructuredAssessmentResult;
  onOpenReviewQueue: () => void;
  onOpenFinalize: () => void;
  onOpenAuditTrail: () => void;
  onOpenRevisionHistory: () => void;
  onOpenAnalytics: () => void;
}

export default function TeacherDashboardOverview({
  structuredResult,
  onOpenReviewQueue,
  onOpenFinalize,
  onOpenAuditTrail,
  onOpenRevisionHistory,
  onOpenAnalytics,
}: Props) {
  const isFinalized = structuredResult.assessment_status === "FINALIZED";
  const pct = structuredResult.percentage;
  const confPct = Math.round(structuredResult.overall_confidence * 100);

  return (
    <div className="bg-white border-b border-slate-200 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
        {/* Header Title Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-slate-900 text-white flex items-center justify-center font-bold text-sm shadow-sm">
              vAI
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold text-slate-900 tracking-tight">
                  Assessment Overview
                </h1>
                <span
                  className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold ${
                    isFinalized
                      ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                      : "bg-amber-100 text-amber-800 border border-amber-300"
                  }`}
                >
                  {isFinalized ? (
                    <>
                      <Lock size={12} /> FINALIZED (Rev #{structuredResult.revision_index})
                    </>
                  ) : (
                    <>
                      <FileCheck size={12} /> IN REVIEW
                    </>
                  )}
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium">
                Assessment ID: <span className="font-mono text-slate-700">{structuredResult.assessment_id}</span>
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={onOpenReviewQueue}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 transition-all ${
                structuredResult.questions_needing_review > 0
                  ? "bg-amber-500 text-white hover:bg-amber-600 shadow-xs animate-pulse"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              <HelpCircle size={14} />
              Review Queue ({structuredResult.questions_needing_review})
            </button>

            <button
              onClick={onOpenAnalytics}
              className="px-3 py-1.5 rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              <BarChart3 size={14} />
              Analytics
            </button>

            <button
              onClick={onOpenAuditTrail}
              className="px-3 py-1.5 rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              <ShieldCheck size={14} />
              Audit Trail ({structuredResult.audit_trail.length})
            </button>

            <button
              onClick={onOpenRevisionHistory}
              className="px-3 py-1.5 rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              <History size={14} />
              Revisions ({structuredResult.version_history.length || 1})
            </button>

            {!isFinalized ? (
              <button
                onClick={onOpenFinalize}
                className="px-4 py-1.5 rounded-xl bg-emerald-600 text-white hover:bg-emerald-700 text-xs font-bold flex items-center gap-1.5 shadow-sm transition-all"
              >
                <FileCheck size={14} />
                Finalize Assessment
              </button>
            ) : (
              <button
                onClick={onOpenFinalize}
                className="px-4 py-1.5 rounded-xl bg-slate-800 text-slate-200 hover:bg-slate-900 text-xs font-bold flex items-center gap-1.5 shadow-sm transition-all"
              >
                <ArrowUpRight size={14} />
                Finalized Details
              </button>
            )}
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {/* Main Score Card */}
          <div className="col-span-2 sm:col-span-1 bg-gradient-to-br from-slate-900 to-slate-800 text-white rounded-2xl p-3.5 shadow-sm">
            <div className="text-[11px] font-medium text-slate-300">Final Score</div>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-black">{structuredResult.final_awarded_marks}</span>
              <span className="text-xs text-slate-400 font-bold">/ {structuredResult.total_max_marks}</span>
            </div>
            <div className="mt-1 flex items-center justify-between text-[11px]">
              <span className="font-extrabold text-emerald-400">{pct}%</span>
              <span className="text-slate-400">Confidence: {confPct}%</span>
            </div>
          </div>

          {/* AI Score */}
          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-3.5">
            <div className="text-[11px] font-semibold text-slate-500">Original AI Score</div>
            <div className="text-xl font-bold text-slate-800 mt-1">
              {structuredResult.ai_awarded_marks} <span className="text-xs font-normal text-slate-500">pts</span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1">Automated Step 4</div>
          </div>

          {/* Teacher Adjusted */}
          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-3.5">
            <div className="text-[11px] font-semibold text-slate-500">Teacher Adjusted</div>
            <div className="text-xl font-bold text-indigo-600 mt-1">
              {structuredResult.teacher_adjusted_marks !== null
                ? `${structuredResult.teacher_adjusted_marks} pts`
                : "No Overrides"}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">3-State Preservation</div>
          </div>

          {/* Questions Needing Review */}
          <div className={`border rounded-2xl p-3.5 ${
            structuredResult.questions_needing_review > 0
              ? "bg-amber-50/60 border-amber-200"
              : "bg-slate-50 border-slate-200/80"
          }`}>
            <div className="text-[11px] font-semibold text-slate-500">Needs Review</div>
            <div className={`text-xl font-bold mt-1 ${
              structuredResult.questions_needing_review > 0 ? "text-amber-700" : "text-slate-700"
            }`}>
              {structuredResult.questions_needing_review} <span className="text-xs font-normal text-slate-500">questions</span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1">Categorized Reasons</div>
          </div>

          {/* Answered / Unanswered */}
          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-3.5">
            <div className="text-[11px] font-semibold text-slate-500">Answered / Total</div>
            <div className="text-xl font-bold text-slate-800 mt-1">
              {structuredResult.answered_questions} <span className="text-xs font-normal text-slate-500">/ {structuredResult.total_questions}</span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              Unanswered: {structuredResult.unanswered_questions}
            </div>
          </div>

          {/* Unmatched Answers */}
          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-3.5">
            <div className="text-[11px] font-semibold text-slate-500">Unmatched Regions</div>
            <div className="text-xl font-bold text-slate-800 mt-1">
              {structuredResult.unmatched_answers_count}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">Step 2 BBoxes</div>
          </div>
        </div>
      </div>
    </div>
  );
}
