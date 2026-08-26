"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Circle,
  ArrowLeft,
  BookOpen,
  Sparkles,
  FileCheck,
  Search,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import AnswerSheetViewer from "@/components/AnswerSheetViewer";
import { getResult, fileUrl } from "@/lib/api";
import { AssessmentResult, QuestionResult, AnswerStatus } from "@/types/assessment";

const STATUS_META: Record<
  AnswerStatus,
  { label: string; icon: typeof CheckCircle2; color: string; bg: string; border: string }
> = {
  matched: {
    label: "Answer Matched",
    icon: CheckCircle2,
    color: "#10b981",
    bg: "#ecfdf5",
    border: "#a7f3d0",
  },
  review_required: {
    label: "Review Recommended",
    icon: HelpCircle,
    color: "#f59e0b",
    bg: "#fffbe6",
    border: "#fde68a",
  },
  unmatched: {
    label: "Uncertain Match",
    icon: AlertTriangle,
    color: "#ef4444",
    bg: "#fef2f2",
    border: "#fecaca",
  },
  unanswered: {
    label: "Unanswered",
    icon: Circle,
    color: "#94a3b8",
    bg: "#f1f5f9",
    border: "#e2e8f0",
  },
};

function confidenceBadge(confidence: number) {
  if (confidence >= 0.85)
    return <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 text-[11px] font-bold">High</span>;
  if (confidence >= 0.55)
    return <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 text-[11px] font-bold">Medium</span>;
  return <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 text-[11px] font-bold">Review</span>;
}

export default function WorkspacePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activePage, setActivePage] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [searchFilter, setSearchFilter] = useState("");
  const [mobileTab, setMobileTab] = useState<"questions" | "sheet" | "insights">("sheet");
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  useEffect(() => {
    getResult(params.id)
      .then((r) => {
        setResult(r);
        if (r.questions.length > 0) setSelectedId(r.questions[0].id);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load result"));
  }, [params.id]);

  const filteredQuestions = useMemo(() => {
    if (!result) return [];
    if (!searchFilter.trim()) return result.questions;
    const q = searchFilter.toLowerCase();
    return result.questions.filter((item) => item.number.toLowerCase().includes(q) || item.text.toLowerCase().includes(q));
  }, [result, searchFilter]);

  const selected: QuestionResult | undefined = useMemo(
    () => result?.questions.find((q) => q.id === selectedId),
    [result, selectedId]
  );

  const answerSheetUrl = fileUrl(params.id, "answer_sheet");

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f4f5f8] p-4 text-center">
        <div className="bg-white rounded-3xl p-8 max-w-md border border-slate-200 shadow-xl">
          <AlertTriangle className="mx-auto text-rose-500 mb-3" size={32} />
          <h2 className="text-lg font-bold text-slate-800 mb-1">Failed to load workspace</h2>
          <p className="text-xs text-slate-500 mb-5">{error}</p>
          <button
            onClick={() => router.push("/")}
            className="px-5 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold hover:bg-[#ff5a1f] transition-colors"
          >
            Return to Upload
          </button>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f4f5f8]">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 border-3 border-[#ff5a1f] border-t-transparent rounded-full animate-spin" />
          <span className="text-xs font-semibold text-slate-600">Loading Assessment Evaluation Workspace...</span>
        </div>
      </div>
    );
  }

  const isPdf = result.answer_sheet_is_pdf;

  return (
    <div className="flex h-screen w-full bg-[#f4f5f8] overflow-hidden font-sans">
      <Sidebar isMobileOpen={isMobileOpen} onMobileClose={() => setIsMobileOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0 h-screen">
        <Header title="Assessment Workspace — Question & Answer Mapping" onMenuClick={() => setIsMobileOpen(true)} />

        {/* Mobile View Tab Selector Bar */}
        <div className="lg:hidden flex items-center justify-around bg-white border-b border-slate-200 px-2 py-2">
          <button
            onClick={() => setMobileTab("questions")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-colors ${
              mobileTab === "questions" ? "bg-[#ff5a1f] text-white" : "text-slate-600 bg-slate-100"
            }`}
          >
            <BookOpen size={14} /> Questions ({result.questions.length})
          </button>
          <button
            onClick={() => setMobileTab("sheet")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-colors ${
              mobileTab === "sheet" ? "bg-[#ff5a1f] text-white" : "text-slate-600 bg-slate-100"
            }`}
          >
            <FileCheck size={14} /> Answer Sheet
          </button>
          <button
            onClick={() => setMobileTab("insights")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-colors ${
              mobileTab === "insights" ? "bg-[#ff5a1f] text-white" : "text-slate-600 bg-slate-100"
            }`}
          >
            <Sparkles size={14} /> AI Insights
          </button>
        </div>

        {/* 3-Column Split View Grid */}
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-[300px_1fr_340px] h-[calc(100vh-4rem)] overflow-hidden">
          {/* Questions Column */}
          <div
            className={`${
              mobileTab === "questions" ? "block" : "hidden"
            } lg:block border-r border-slate-200 bg-white flex flex-col h-full overflow-hidden`}
          >
            <div className="p-3.5 border-b border-slate-200 bg-slate-50/50 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                  Extracted Questions
                </span>
                <span className="px-2 py-0.5 rounded-full bg-slate-200 text-slate-700 text-[11px] font-bold">
                  {result.questions.length} total
                </span>
              </div>

              <div className="relative">
                <Search size={13} className="absolute left-3 top-2.5 text-slate-400" />
                <input
                  type="text"
                  placeholder="Filter question number or text..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 rounded-xl bg-white border border-slate-200 text-xs text-slate-700 placeholder-slate-400 focus:outline-none focus:border-[#ff5a1f]"
                />
              </div>
            </div>

            <div className="flex-1 overflow-y-auto scrollbar-thin p-2 space-y-1.5">
              {filteredQuestions.map((q) => {
                const meta = STATUS_META[q.answer.status];
                const Icon = meta.icon;
                const active = q.id === selectedId;

                return (
                  <button
                    key={q.id}
                    onClick={() => {
                      setSelectedId(q.id);
                      setMobileTab("sheet");
                    }}
                    className={`w-full text-left rounded-2xl p-3 border transition-all duration-200 cursor-pointer ${
                      active
                        ? "border-[#ff5a1f] bg-[#fff0e8]/60 shadow-xs ring-1 ring-[#ff5a1f]/30"
                        : "border-slate-100 bg-slate-50/50 hover:bg-slate-100/80 hover:border-slate-200"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-xs font-bold px-2 py-0.5 rounded-lg ${
                            active ? "bg-[#ff5a1f] text-white" : "bg-slate-200 text-slate-700"
                          }`}
                        >
                          Q{q.number}
                        </span>
                        {q.answer.status !== "unanswered" && confidenceBadge(q.answer.confidence)}
                      </div>
                      <Icon size={16} style={{ color: meta.color }} />
                    </div>
                    <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed font-medium">{q.text}</p>
                  </button>
                );
              })}

              {result.unmatched_answers.length > 0 && (
                <div className="mt-4 p-3 rounded-2xl bg-amber-50 border border-amber-200">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-amber-800 mb-1">
                    <AlertTriangle size={14} className="text-amber-600" />
                    <span>Unmatched Answers ({result.unmatched_answers.length})</span>
                  </div>
                  <p className="text-[11px] text-amber-700 leading-normal">
                    Student text recognized on the sheet that did not meet the confidence threshold for auto-linking.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Answer Sheet Viewer Column */}
          <div
            className={`${
              mobileTab === "sheet" ? "block" : "hidden"
            } lg:block border-r border-slate-200 bg-[#eef0f4] h-full overflow-hidden`}
          >
            <AnswerSheetViewer
              fileUrl={answerSheetUrl}
              isPdf={isPdf}
              regions={selected?.answer.regions ?? []}
              pageSizes={result.answer_sheet_page_sizes}
              activePage={activePage}
              onPageChange={setActivePage}
              totalPages={result.answer_sheet_pages}
            />
          </div>

          {/* AI Insights Column */}
          <div
            className={`${
              mobileTab === "insights" ? "block" : "hidden"
            } lg:block bg-white h-full overflow-y-auto scrollbar-thin p-5`}
          >
            {selected ? (
              <div className="space-y-5">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                      Question {selected.number} Details
                    </span>
                    {confidenceBadge(selected.answer.confidence)}
                  </div>
                  <h2 className="text-sm font-bold text-slate-900 leading-snug">{selected.text}</h2>
                </div>

                <div
                  className="rounded-2xl p-4 border flex items-center gap-3"
                  style={{
                    backgroundColor: STATUS_META[selected.answer.status].bg,
                    borderColor: STATUS_META[selected.answer.status].border,
                  }}
                >
                  {(() => {
                    const Icon = STATUS_META[selected.answer.status].icon;
                    return <Icon size={20} style={{ color: STATUS_META[selected.answer.status].color }} />;
                  })()}
                  <div>
                    <div className="text-xs font-bold text-slate-900">{STATUS_META[selected.answer.status].label}</div>
                    {selected.answer.method && (
                      <div className="text-[11px] text-slate-500 capitalize">
                        Method: {selected.answer.method.replaceAll("_", " ")}
                      </div>
                    )}
                  </div>
                </div>

                {selected.answer.status !== "unanswered" ? (
                  <div className="rounded-2xl border border-slate-200 bg-slate-50/60 p-4 space-y-2">
                    <div className="flex items-center justify-between text-xs font-bold text-slate-700">
                      <span>Extracted Student Response</span>
                      <span className="text-[#ff5a1f] font-mono">
                        {Math.round(selected.answer.confidence * 100)}% Match
                      </span>
                    </div>
                    <p className="text-xs text-slate-800 leading-relaxed font-mono bg-white p-3 rounded-xl border border-slate-200 whitespace-pre-wrap">
                      {selected.answer.text}
                    </p>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-500 text-center">
                    No answer was recognized for this question on the submitted sheet.
                  </div>
                )}

                {selected.grading && (
                  <div className="rounded-2xl border border-purple-200 bg-purple-50/50 p-4 space-y-2">
                    <div className="flex items-center gap-2 text-xs font-bold text-purple-900">
                      <Sparkles size={15} className="text-purple-600" />
                      <span>AI Teacher Feedback &amp; Score</span>
                    </div>
                    <p className="text-xs text-purple-800 leading-relaxed">{selected.grading.feedback}</p>
                  </div>
                )}

                <button
                  onClick={() => router.push("/")}
                  className="w-full flex items-center justify-center gap-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 py-2.5 text-xs font-bold transition-colors cursor-pointer"
                >
                  <ArrowLeft size={14} /> Back to Dashboard
                </button>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center text-slate-400 p-6">
                <BookOpen size={32} className="mb-2 text-slate-300" />
                <p className="text-xs font-medium">Select a question from the left sidebar to inspect extracted answers and mapping regions.</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
