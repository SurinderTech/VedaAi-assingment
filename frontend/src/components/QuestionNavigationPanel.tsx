"use client";

import { useMemo, useState } from "react";
import { StructuredQuestionResult } from "@/types/assessment";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Circle,
  Search,
  PenTool,
  Filter,
} from "lucide-react";

interface Props {
  questions: StructuredQuestionResult[];
  selectedId: string | null;
  onSelectQuestion: (questionId: string) => void;
}

type FilterTab = "ALL" | "NEEDS_REVIEW" | "UNANSWERED" | "OVERRIDDEN" | "HIGH_CONFIDENCE";

export default function QuestionNavigationPanel({
  questions,
  selectedId,
  onSelectQuestion,
}: Props) {
  const [filterTab, setFilterTab] = useState<FilterTab>("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredQuestions = useMemo(() => {
    let list = questions;

    if (filterTab === "NEEDS_REVIEW") {
      list = list.filter((q) => q.needs_review || q.review_status === "PENDING_REVIEW");
    } else if (filterTab === "UNANSWERED") {
      list = list.filter((q) => q.status === "unanswered");
    } else if (filterTab === "OVERRIDDEN") {
      list = list.filter((q) => q.teacher_adjusted_marks !== null || q.review_status === "TEACHER_OVERRIDE");
    } else if (filterTab === "HIGH_CONFIDENCE") {
      list = list.filter((q) => q.evaluation_confidence >= 0.85 && !q.needs_review);
    }

    if (searchQuery.trim()) {
      const sq = searchQuery.toLowerCase();
      list = list.filter(
        (q) =>
          q.question_number.toLowerCase().includes(sq) ||
          q.question_text.toLowerCase().includes(sq)
      );
    }

    return list;
  }, [questions, filterTab, searchQuery]);

  return (
    <div className="w-80 bg-white border-r border-slate-200 flex flex-col h-full shrink-0">
      {/* Header & Search */}
      <div className="p-3 border-b border-slate-200 bg-slate-50/50">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 text-slate-400" size={14} />
          <input
            type="text"
            placeholder="Search questions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 bg-white border border-slate-200 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:outline-hidden focus:border-slate-400 focus:ring-1 focus:ring-slate-400 transition-all font-medium"
          />
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-1 mt-2.5 overflow-x-auto no-scrollbar pb-1 text-[11px] font-semibold">
          {(
            [
              { key: "ALL", label: "All" },
              { key: "NEEDS_REVIEW", label: "Review" },
              { key: "UNANSWERED", label: "Unanswered" },
              { key: "OVERRIDDEN", label: "Overridden" },
              { key: "HIGH_CONFIDENCE", label: "High Conf" },
            ] as const
          ).map((tab) => (
            <button
              key={tab.key}
              onClick={() => setFilterTab(tab.key)}
              className={`px-2.5 py-1 rounded-lg shrink-0 transition-all ${
                filterTab === tab.key
                  ? "bg-slate-900 text-white shadow-xs"
                  : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200/60"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Question List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {filteredQuestions.length === 0 ? (
          <div className="p-6 text-center text-xs text-slate-400">
            No questions match current filter
          </div>
        ) : (
          filteredQuestions.map((q) => {
            const isSelected = q.question_id === selectedId || q.question_number === selectedId;
            const isOverridden = q.teacher_adjusted_marks !== null || q.review_status === "TEACHER_OVERRIDE";
            const isUnanswered = q.status === "unanswered";
            const isNeedsReview = q.needs_review || q.review_status === "PENDING_REVIEW";

            let icon = <CheckCircle2 size={14} className="text-emerald-500 shrink-0" />;
            let statusLabel = "Graded";
            let statusBg = "bg-emerald-50 text-emerald-700 border-emerald-200";

            if (isOverridden) {
              icon = <PenTool size={14} className="text-indigo-600 shrink-0" />;
              statusLabel = "Overridden";
              statusBg = "bg-indigo-50 text-indigo-700 border-indigo-200";
            } else if (isUnanswered) {
              icon = <Circle size={14} className="text-slate-400 shrink-0" />;
              statusLabel = "Unanswered";
              statusBg = "bg-slate-100 text-slate-600 border-slate-200";
            } else if (isNeedsReview) {
              icon = <HelpCircle size={14} className="text-amber-500 shrink-0" />;
              statusLabel = "Needs Review";
              statusBg = "bg-amber-50 text-amber-700 border-amber-200 animate-pulse";
            }

            return (
              <button
                key={q.question_id}
                onClick={() => onSelectQuestion(q.question_id)}
                className={`w-full text-left p-2.5 rounded-xl border transition-all flex items-start justify-between gap-2 ${
                  isSelected
                    ? "bg-slate-900 text-white border-slate-900 shadow-sm"
                    : "bg-white hover:bg-slate-50 border-slate-200/80 text-slate-800"
                }`}
              >
                <div className="flex items-start gap-2 min-w-0">
                  <div className="mt-0.5">{icon}</div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className={`font-bold text-xs ${isSelected ? "text-white" : "text-slate-900"}`}>
                        Q{q.question_number}
                      </span>
                      <span className={`text-[10px] px-1.5 py-0.2 rounded-md font-semibold border ${
                        isSelected ? "bg-slate-800 text-slate-200 border-slate-700" : statusBg
                      }`}>
                        {statusLabel}
                      </span>
                    </div>
                    <p className={`text-[11px] truncate mt-0.5 ${isSelected ? "text-slate-300" : "text-slate-500"}`}>
                      {q.question_text || "No text available"}
                    </p>
                  </div>
                </div>

                <div className="text-right shrink-0">
                  <div className={`font-extrabold text-xs ${isSelected ? "text-emerald-400" : "text-slate-900"}`}>
                    {q.awarded_marks} / {q.max_marks}
                  </div>
                  <div className={`text-[10px] font-mono ${isSelected ? "text-slate-400" : "text-slate-400"}`}>
                    {Math.round(q.evaluation_confidence * 100)}% conf
                  </div>
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
