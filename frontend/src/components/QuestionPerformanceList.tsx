"use client";

import { useState } from "react";
import { QuestionPerformanceSummary } from "@/types/assessment";
import { CheckCircle2, Circle, ChevronDown, ChevronUp, Sparkles, AlertCircle } from "lucide-react";

interface Props {
  questions: QuestionPerformanceSummary[];
}

export default function QuestionPerformanceList({ questions }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(
    questions.length > 0 ? questions[0].question_id : null
  );

  const toggleExpand = (qId: string) => {
    setExpandedId(expandedId === qId ? null : qId);
  };

  return (
    <div className="bg-white rounded-3xl border border-slate-200 p-5 space-y-4 shadow-xs">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-black text-slate-900">
          Question-by-Question Performance ({questions.length})
        </h2>
        <span className="text-xs text-slate-400 font-medium">Grounded Evidence Results</span>
      </div>

      <div className="space-y-3">
        {questions.map((q) => {
          const isExpanded = expandedId === q.question_id;
          const isUnanswered = q.status === "unanswered";

          return (
            <div
              key={q.question_id}
              className="border border-slate-200 rounded-2xl overflow-hidden transition-all bg-white"
            >
              <button
                onClick={() => toggleExpand(q.question_id)}
                className="w-full p-4 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className="font-extrabold text-xs text-slate-900 bg-slate-100 border border-slate-200 px-2.5 py-1 rounded-lg shrink-0">
                    Q{q.question_number}
                  </span>
                  <div className="min-w-0">
                    <p className="text-xs font-bold text-slate-800 truncate">
                      {q.question_text || "Question prompt"}
                    </p>
                    <div className="flex items-center gap-2 mt-0.5 text-[11px] text-slate-500">
                      <span className={`px-2 py-0.2 rounded-md font-semibold border ${
                        isUnanswered ? "bg-slate-100 text-slate-600 border-slate-200" : "bg-emerald-50 text-emerald-700 border-emerald-200"
                      }`}>
                        {isUnanswered ? "Unanswered" : "Graded"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0 ml-2">
                  <div className="text-right">
                    <span className="text-sm font-black text-slate-900">
                      {q.final_awarded_marks} / {q.max_marks} pts
                    </span>
                    <div className="text-[10px] font-bold text-emerald-600">
                      {q.percentage}%
                    </div>
                  </div>
                  {isExpanded ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
                </div>
              </button>

              {isExpanded && (
                <div className="p-4 bg-slate-50 border-t border-slate-200 space-y-3 text-xs">
                  {q.feedback && (
                    <div className="p-3 bg-white border border-slate-200 rounded-xl text-slate-700 font-medium leading-relaxed">
                      <span className="font-bold text-slate-900 block mb-1">Feedback:</span>
                      {q.feedback}
                    </div>
                  )}

                  {q.strengths.length > 0 && (
                    <div className="space-y-1">
                      <span className="font-bold text-emerald-800 text-[11px]">Strengths Demonstrated:</span>
                      <ul className="list-disc list-inside space-y-0.5 text-slate-600 text-[11px]">
                        {q.strengths.map((s, idx) => (
                          <li key={idx}>{s}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {q.improvement_points.length > 0 && (
                    <div className="space-y-1">
                      <span className="font-bold text-amber-800 text-[11px]">Improvement Opportunities:</span>
                      <ul className="list-disc list-inside space-y-0.5 text-slate-600 text-[11px]">
                        {q.improvement_points.map((imp, idx) => (
                          <li key={idx}>{imp}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {q.criteria_summary.length > 0 && (
                    <div className="pt-2 border-t border-slate-200/80 space-y-1">
                      <span className="font-bold text-slate-900 text-[11px]">Rubric Criteria Mastery:</span>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-1">
                        {q.criteria_summary.map((c) => (
                          <div key={c.criterion_id} className="p-2 bg-white border border-slate-200 rounded-xl text-[11px] space-y-0.5">
                            <div className="flex justify-between font-bold text-slate-800">
                              <span>{c.description}</span>
                              <span className="text-indigo-600">{c.awarded_marks}/{c.max_marks} pts</span>
                            </div>
                            <div className="text-slate-400 text-[10px]">Status: <span className="font-semibold text-slate-600">{c.evidence_status}</span></div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
