"use client";

import { useEffect, useState } from "react";
import { AssessmentInsights, AssessmentInsight } from "@/types/assessment";
import { getAssessmentInsights } from "@/lib/api";
import { Sparkles, CheckCircle2, AlertTriangle, AlertCircle, Eye, ChevronRight } from "lucide-react";

interface Props {
  assessmentId: string;
  onSelectQuestion: (questionId: string) => void;
}

export default function AssessmentInsightsPanel({ assessmentId, onSelectQuestion }: Props) {
  const [insights, setInsights] = useState<AssessmentInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!assessmentId) return;
    setLoading(true);
    getAssessmentInsights(assessmentId)
      .then((data) => {
        setInsights(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load insights");
        setLoading(false);
      });
  }, [assessmentId]);

  if (loading) {
    return (
      <div className="bg-white border border-slate-200 rounded-3xl p-5 shadow-xs flex items-center justify-center min-h-[140px]">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-500">
          <div className="h-4 w-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
          Generating Assessment Intelligence & Teacher Insights...
        </div>
      </div>
    );
  }

  if (error || !insights) {
    return null; // Graceful failure isolation
  }

  return (
    <div className="bg-white border border-slate-200 rounded-3xl p-5 space-y-4 shadow-xs">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold">
            <Sparkles size={18} />
          </div>
          <div>
            <h3 className="text-sm font-black text-slate-900 tracking-tight">Assessment Intelligence Insights</h3>
            <p className="text-[11px] text-slate-500 font-medium">Evidence-grounded teacher decision support</p>
          </div>
        </div>
        <span className="text-xs font-extrabold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-full border border-indigo-100">
          {insights.final_awarded_marks} / {insights.total_max_marks} pts ({insights.percentage}%)
        </span>
      </div>

      {/* Review Priorities (If Any) */}
      {insights.review_priorities.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
            <Eye size={14} className="text-amber-600" /> Review Priorities ({insights.review_priorities.length})
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            {insights.review_priorities.map((item) => (
              <button
                key={item.insight_id}
                onClick={() => item.question_ids[0] && onSelectQuestion(item.question_ids[0])}
                className="p-3 bg-amber-50/70 border border-amber-200/90 rounded-2xl text-left hover:bg-amber-100/80 transition-all flex items-start justify-between group"
              >
                <div>
                  <div className="font-bold text-amber-900 text-xs flex items-center gap-1">
                    {item.title}
                  </div>
                  <p className="text-[11px] text-amber-800/90 mt-0.5 font-medium leading-snug">{item.summary}</p>
                </div>
                <ChevronRight size={14} className="text-amber-500 group-hover:translate-x-0.5 transition-transform shrink-0 mt-0.5" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error Patterns (If Any) */}
      {insights.error_patterns.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
            <AlertTriangle size={14} className="text-rose-600" /> Observed Error Patterns ({insights.error_patterns.length})
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            {insights.error_patterns.map((errItem) => (
              <button
                key={errItem.insight_id}
                onClick={() => errItem.question_ids[0] && onSelectQuestion(errItem.question_ids[0])}
                className="p-3 bg-rose-50/70 border border-rose-200/90 rounded-2xl text-left hover:bg-rose-100/80 transition-all flex items-start justify-between group"
              >
                <div>
                  <div className="font-bold text-rose-900 text-xs flex items-center gap-1">
                    {errItem.title}
                  </div>
                  <p className="text-[11px] text-rose-800/90 mt-0.5 font-medium leading-snug">{errItem.summary}</p>
                </div>
                <ChevronRight size={14} className="text-rose-500 group-hover:translate-x-0.5 transition-transform shrink-0 mt-0.5" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Strengths & Attention Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1 text-xs">
        {insights.strengths.length > 0 && (
          <div className="p-3 bg-emerald-50/60 border border-emerald-200/80 rounded-2xl space-y-1">
            <span className="font-bold text-emerald-900 flex items-center gap-1 text-[11px]">
              <CheckCircle2 size={13} className="text-emerald-600" /> Strongest Demonstrated Areas
            </span>
            <ul className="list-disc list-inside space-y-0.5 text-[11px] text-emerald-800 font-medium">
              {insights.strengths.slice(0, 3).map((s, idx) => (
                <li key={idx} className="truncate">{s}</li>
              ))}
            </ul>
          </div>
        )}

        {insights.areas_needing_attention.length > 0 && (
          <div className="p-3 bg-slate-50 border border-slate-200/80 rounded-2xl space-y-1">
            <span className="font-bold text-slate-800 flex items-center gap-1 text-[11px]">
              <AlertCircle size={13} className="text-slate-600" /> Areas Needing Attention
            </span>
            <ul className="list-disc list-inside space-y-0.5 text-[11px] text-slate-700 font-medium">
              {insights.areas_needing_attention.slice(0, 3).map((a, idx) => (
                <li key={idx} className="truncate">{a}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
