"use client";

import { useMemo, useState } from "react";
import {
  StructuredQuestionResult,
  StructuredAssessmentResult,
  UnmatchedAnswer,
} from "@/types/assessment";
import { overrideQuestionMarks } from "@/lib/api";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Circle,
  Search,
  Sparkles,
  ChevronDown,
  ChevronUp,
  FileText,
  PenTool,
  Save,
  ShieldCheck,
  BookOpen,
} from "lucide-react";

interface Props {
  assessmentId: string;
  structuredResult: StructuredAssessmentResult;
  unmatchedAnswers?: UnmatchedAnswer[];
  selectedId: string | null;
  onSelectQuestion: (questionId: string) => void;
  onSelectUnmatched?: (unmatched: UnmatchedAnswer) => void;
  onStructuredResultUpdated: (updated: StructuredAssessmentResult) => void;
}

type FilterTab = "ALL" | "NEEDS_REVIEW" | "UNANSWERED" | "OVERRIDDEN" | "UNMATCHED";

export default function ExtractedQuestionsPanel({
  assessmentId,
  structuredResult,
  unmatchedAnswers = [],
  selectedId,
  onSelectQuestion,
  onSelectUnmatched,
  onStructuredResultUpdated,
}: Props) {
  const [filterTab, setFilterTab] = useState<FilterTab>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(selectedId);

  // Form states for Teacher Override on active question
  const [teacherMarksInput, setTeacherMarksInput] = useState<string>("");
  const [criterionInputs, setCriterionInputs] = useState<Record<string, string>>({});
  const [reasonInput, setReasonInput] = useState("");
  const [commentInput, setCommentInput] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const isFinalized = structuredResult.assessment_status === "FINALIZED";
  const questions = structuredResult.question_results;

  // Selected Question object
  const selectedQuestion = useMemo(() => {
    return questions.find(
      (q) => q.question_id === selectedId || q.question_number === selectedId
    );
  }, [questions, selectedId]);

  // Sync inputs when selectedQuestion changes
  const handleSelectQuestion = (qId: string) => {
    onSelectQuestion(qId);
    setExpandedId(qId);
    setErrorMsg(null);
    setSuccessMsg(null);

    const q = questions.find((item) => item.question_id === qId || item.question_number === qId);
    if (q) {
      setTeacherMarksInput(
        q.teacher_adjusted_marks !== null
          ? String(q.teacher_adjusted_marks)
          : String(q.awarded_marks)
      );
      const initialCr: Record<string, string> = {};
      q.criterion_results.forEach((c) => {
        initialCr[c.criterion_id] = String(c.awarded_marks);
      });
      setCriterionInputs(initialCr);
    }
  };

  const handleToggleExpand = (e: React.MouseEvent, qId: string) => {
    e.stopPropagation();
    if (expandedId === qId) {
      setExpandedId(null);
    } else {
      handleSelectQuestion(qId);
    }
  };

  // Filtered list logic
  const filteredQuestions = useMemo(() => {
    let list = questions;

    if (filterTab === "NEEDS_REVIEW") {
      list = list.filter((q) => q.needs_review || q.review_status === "PENDING_REVIEW");
    } else if (filterTab === "UNANSWERED") {
      list = list.filter((q) => q.status === "unanswered");
    } else if (filterTab === "OVERRIDDEN") {
      list = list.filter(
        (q) => q.teacher_adjusted_marks !== null || q.review_status === "TEACHER_OVERRIDE"
      );
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

  const handleCriterionChange = (criterionId: string, val: string, question: StructuredQuestionResult) => {
    const updated = { ...criterionInputs, [criterionId]: val };
    setCriterionInputs(updated);

    let sum = 0;
    let validSum = true;
    question.criterion_results.forEach((c) => {
      const inputStr = updated[c.criterion_id] ?? String(c.awarded_marks);
      const parsed = parseFloat(inputStr);
      if (!isNaN(parsed)) {
        sum += parsed;
      } else {
        validSum = false;
      }
    });
    if (validSum) {
      setTeacherMarksInput(String(Math.min(question.max_marks, sum)));
    }
  };

  const handleSaveOverride = async (question: StructuredQuestionResult) => {
    setIsSubmitting(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    const parsedScore = parseFloat(teacherMarksInput);
    if (isNaN(parsedScore) || parsedScore < 0 || parsedScore > question.max_marks) {
      setErrorMsg(`Marks must be between 0.0 and ${question.max_marks}`);
      setIsSubmitting(false);
      return;
    }

    const crOverrides: Record<string, number> = {};
    if (question.criterion_results.length > 0) {
      for (const c of question.criterion_results) {
        const valStr = criterionInputs[c.criterion_id];
        const pVal = parseFloat(valStr);
        if (isNaN(pVal) || pVal < 0 || pVal > c.max_marks) {
          setErrorMsg(`Criterion '${c.description}' marks must be between 0.0 and ${c.max_marks}`);
          setIsSubmitting(false);
          return;
        }
        crOverrides[c.criterion_id] = pVal;
      }
    }

    try {
      const updated = await overrideQuestionMarks(assessmentId, question.question_id, {
        teacher_marks: parsedScore,
        criterion_overrides: Object.keys(crOverrides).length > 0 ? crOverrides : undefined,
        comment: commentInput.trim() || undefined,
        reason: reasonInput.trim() || "Teacher mark override",
        reviewer: "Teacher",
      });
      setSuccessMsg("Override saved!");
      onStructuredResultUpdated(updated);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Override failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="w-full h-full bg-[#f8fafc] border-r border-slate-200/90 flex flex-col overflow-hidden">
      {/* Header & Filter Controls */}
      <div className="p-3.5 bg-white border-b border-slate-200 shadow-2xs space-y-2.5 shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-7 w-7 rounded-lg bg-slate-900 text-white flex items-center justify-center font-bold text-xs shadow-xs">
              Q
            </div>
            <h2 className="text-sm font-extrabold text-slate-900 tracking-tight">
              Extracted Questions
            </h2>
          </div>
          <span className="text-[11px] font-mono font-semibold px-2 py-0.5 bg-slate-100 border border-slate-200 text-slate-600 rounded-full">
            {questions.length} Question{questions.length !== 1 ? "s" : ""}
          </span>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-2.5 text-slate-400" size={14} />
          <input
            type="text"
            placeholder="Search questions or keyword..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-hidden focus:border-slate-400 transition-all font-medium"
          />
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-1 overflow-x-auto no-scrollbar text-[11px] font-semibold pt-0.5">
          {(
            [
              { key: "ALL", label: "All Questions" },
              { key: "NEEDS_REVIEW", label: `Review (${structuredResult.questions_needing_review})` },
              { key: "UNANSWERED", label: `Unanswered (${structuredResult.unanswered_questions})` },
              { key: "OVERRIDDEN", label: "Overridden" },
              { key: "UNMATCHED", label: `Unmatched (${unmatchedAnswers.length || structuredResult.unmatched_answers_count})` },
            ] as const
          ).map((tab) => {
            const isActive = filterTab === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => setFilterTab(tab.key)}
                className={`px-2.5 py-1 rounded-lg shrink-0 transition-all cursor-pointer ${
                  isActive
                    ? "bg-slate-900 text-white shadow-xs"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200 border border-slate-200/60"
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Question List Area */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
        {filterTab === "UNMATCHED" ? (
          /* Unmatched Answer Regions List */
          <div className="space-y-2">
            <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-800 space-y-1">
              <div className="font-bold flex items-center gap-1.5">
                <AlertTriangle size={14} className="text-amber-600" /> Unmatched Student Answer Regions
              </div>
              <p className="text-[11px] text-amber-700 leading-relaxed">
                These student handwriting regions were extracted from the answer sheet during OCR but were not confidently matched to any question by Step 3 mapping.
              </p>
            </div>

            {unmatchedAnswers.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-400 bg-white rounded-2xl border border-slate-200">
                No unmatched answer regions detected
              </div>
            ) : (
              unmatchedAnswers.map((unm, idx) => (
                <div
                  key={unm.answer_id || idx}
                  onClick={() => onSelectUnmatched && onSelectUnmatched(unm)}
                  className="p-3 bg-white border border-amber-200 rounded-2xl shadow-2xs hover:border-amber-400 transition-all cursor-pointer space-y-1.5"
                >
                  <div className="flex items-center justify-between text-xs font-bold text-slate-900">
                    <span className="flex items-center gap-1.5 text-amber-700">
                      <FileText size={14} /> Unmatched Region #{idx + 1}
                    </span>
                    <span className="text-[10px] font-mono bg-amber-100 text-amber-800 px-2 py-0.5 rounded-md">
                      Conf: {Math.round(unm.confidence * 100)}%
                    </span>
                  </div>
                  <p className="text-xs font-mono text-slate-700 bg-slate-50 p-2 rounded-xl border border-slate-200">
                    {unm.text || "Unrecognized handwriting fragment"}
                  </p>
                </div>
              ))
            )}
          </div>
        ) : filteredQuestions.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-400 bg-white rounded-2xl border border-slate-200">
            No questions match current filter criteria
          </div>
        ) : (
          filteredQuestions.map((q) => {
            const isSelected = q.question_id === selectedId || q.question_number === selectedId;
            const isExpanded = expandedId === q.question_id || expandedId === q.question_number;
            const isUnanswered = q.status === "unanswered";
            const isOverridden = q.teacher_adjusted_marks !== null || q.review_status === "TEACHER_OVERRIDE";
            const isNeedsReview = q.needs_review || q.review_status === "PENDING_REVIEW";

            // Score Pill Color Logic
            let scorePillBg = "bg-emerald-100 text-emerald-800 border-emerald-300";
            if (isUnanswered) {
              scorePillBg = "bg-slate-100 text-slate-600 border-slate-300";
            } else if (q.awarded_marks === 0) {
              scorePillBg = "bg-rose-100 text-rose-800 border-rose-300";
            } else if (q.awarded_marks < q.max_marks) {
              scorePillBg = "bg-amber-100 text-amber-800 border-amber-300";
            }

            return (
              <div
                key={q.question_id}
                onClick={() => handleSelectQuestion(q.question_id)}
                className={`rounded-2xl border transition-all cursor-pointer overflow-hidden ${
                  isSelected
                    ? "bg-white border-slate-900 shadow-md ring-2 ring-slate-900/10"
                    : "bg-white border-slate-200/90 hover:border-slate-300 shadow-2xs"
                }`}
              >
                {/* Main Collapsed Header matching Figma screenshot */}
                <div className="p-3.5 flex items-start gap-3 justify-between">
                  <div className="flex items-start gap-3 min-w-0">
                    {/* Dark Filled Circle Badge for Question Number */}
                    <div
                      className={`h-7 w-7 rounded-full flex items-center justify-center text-xs font-black shrink-0 shadow-2xs ${
                        isSelected
                          ? "bg-slate-900 text-white ring-2 ring-slate-900/30"
                          : "bg-slate-800 text-slate-100"
                      }`}
                    >
                      {q.question_number}
                    </div>

                    <div className="min-w-0 pt-0.5 space-y-1">
                      <p className="text-xs text-slate-800 font-semibold leading-relaxed line-clamp-3">
                        {q.question_text || "Question prompt text"}
                      </p>

                      {/* MCQ Inline Options Preview (collapsed) — Fix 5 */}
                      {q.options && q.options.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-0.5">
                          {q.options.slice(0, 4).map((opt, idx) => (
                            <span
                              key={idx}
                              className="text-[10px] font-mono bg-blue-50 text-blue-700 border border-blue-200 px-1.5 py-0.5 rounded-md leading-tight"
                            >
                              {opt.length > 20 ? opt.slice(0, 20) + "…" : opt}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Needs Review or Overridden Badges */}
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {/* Question type badge */}
                        {q.question_type && q.question_type !== "UNKNOWN" && (
                          <span className="text-[10px] font-bold px-2 py-0.2 rounded-md bg-sky-50 text-sky-700 border border-sky-200">
                            {q.question_type === "MCQ" ? "MCQ" : q.question_type}
                          </span>
                        )}
                        {isOverridden && (
                          <span className="text-[10px] font-bold px-2 py-0.2 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200 flex items-center gap-1">
                            <PenTool size={10} /> Overridden
                          </span>
                        )}
                        {isNeedsReview && !isOverridden && (
                          <span className="text-[10px] font-bold px-2 py-0.2 rounded-md bg-amber-50 text-amber-800 border border-amber-300 flex items-center gap-1 animate-pulse">
                            <HelpCircle size={10} /> Needs Review
                          </span>
                        )}
                        {isUnanswered && (
                          <span className="text-[10px] font-bold px-2 py-0.2 rounded-md bg-slate-100 text-slate-600 border border-slate-200">
                            No answer detected
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Right Score Pill & Expand Toggle */}
                  <div className="flex items-center gap-2 shrink-0 pt-0.5">
                    <span
                      className={`text-xs font-black px-2.5 py-1 rounded-full border shadow-2xs ${scorePillBg}`}
                    >
                      {q.awarded_marks} / {q.max_marks}
                    </span>

                    <button
                      onClick={(e) => handleToggleExpand(e, q.question_id)}
                      className="p-1 text-slate-400 hover:text-slate-700 rounded-lg hover:bg-slate-100 transition-colors"
                      title={isExpanded ? "Collapse" : "Expand"}
                    >
                      {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                  </div>
                </div>

                {/* Expanded Details Body matching assignment spec */}
                {isExpanded && (
                  <div className="px-4 pb-4 pt-1 border-t border-slate-100 bg-slate-50/40 space-y-3.5 text-xs">
                    {/* AI Feedback Banner */}
                    <div className="bg-white p-3.5 rounded-xl border border-slate-200/90 space-y-1.5 shadow-2xs">
                      <div className="flex items-center gap-1.5 font-bold text-slate-900 text-[11px]">
                        <Sparkles size={14} className="text-amber-500" /> AI Feedback
                      </div>
                      <p className="text-slate-700 text-xs leading-relaxed font-medium">
                        {q.feedback ? (
                          q.feedback
                        ) : (
                          <span className="text-slate-400 italic">No feedback available</span>
                        )}
                      </p>
                    </div>

                    {/* Extracted Student Answer Text */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between font-bold text-slate-800 text-[11px]">
                        <span className="flex items-center gap-1.5">
                          <FileText size={14} className="text-slate-500" /> Extracted Student Answer
                        </span>
                        {q.answer_pages && q.answer_pages.length > 0 && (
                          <span className="text-[10px] font-mono text-slate-500">
                            Page {q.answer_pages.join(", ")}
                          </span>
                        )}
                      </div>

                      <div className="p-3 bg-slate-900 text-slate-100 font-mono text-[11.5px] rounded-xl leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto">
                        {q.answer_text ? (
                          q.answer_text
                        ) : (
                          <span className="text-slate-400 italic font-sans text-xs">
                            No answer detected (Unanswered)
                          </span>
                        )}
                      </div>
                    </div>

                    {/* MCQ Options Block (Expanded) — Fix 5 */}
                    {q.extracted_options && q.extracted_options.length > 0 ? (
                      <div className="space-y-1.5">
                        <div className="font-bold text-slate-800 text-[11px] flex items-center gap-1.5">
                          <BookOpen size={13} className="text-blue-500" />
                          MCQ Options ({q.extracted_options.length})
                        </div>
                        <div className="grid grid-cols-2 gap-1.5">
                          {q.extracted_options.map((opt, idx) => (
                            <div
                              key={opt.option_id || idx}
                              className="flex items-start gap-2 p-2 bg-white border border-slate-200 rounded-xl"
                            >
                              <span className="h-5 w-5 rounded-md bg-slate-800 text-white text-[10px] font-black flex items-center justify-center shrink-0">
                                {opt.label || String.fromCharCode(65 + idx)}
                              </span>
                              <span className="text-[11px] text-slate-700 leading-snug font-medium">
                                {opt.full_text || opt.text || ""}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : q.options && q.options.length > 0 ? (
                      <div className="space-y-1.5">
                        <div className="font-bold text-slate-800 text-[11px] flex items-center gap-1.5">
                          <BookOpen size={13} className="text-blue-500" />
                          MCQ Options ({q.options.length})
                        </div>
                        <div className="grid grid-cols-2 gap-1.5">
                          {q.options.map((opt, idx) => {
                            // Parse "A. Text" or "A) Text" format
                            const m = opt.match(/^\s*([A-Da-d])[.)\s]+(.*)$/);
                            const label = m ? m[1].toUpperCase() : String.fromCharCode(65 + idx);
                            const text = m ? m[2].trim() : opt;
                            return (
                              <div
                                key={idx}
                                className="flex items-start gap-2 p-2 bg-white border border-slate-200 rounded-xl"
                              >
                                <span className="h-5 w-5 rounded-md bg-slate-800 text-white text-[10px] font-black flex items-center justify-center shrink-0">
                                  {label}
                                </span>
                                <span className="text-[11px] text-slate-700 leading-snug font-medium">
                                  {text}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : null}

                    {/* Rubric Criteria Evaluation */}
                    {q.criterion_results && q.criterion_results.length > 0 && (
                      <div className="space-y-1.5">
                        <div className="font-bold text-slate-800 text-[11px] flex items-center justify-between">
                          <span>Rubric Criteria Evaluation ({q.criterion_results.length})</span>
                        </div>

                        <div className="space-y-1.5">
                          {q.criterion_results.map((c) => {
                            let badgeStyle = "bg-emerald-50 text-emerald-700 border-emerald-200";
                            let icon = <CheckCircle2 size={13} className="text-emerald-600 shrink-0" />;

                            if (c.evidence_status === "partially_present") {
                              badgeStyle = "bg-amber-50 text-amber-800 border-amber-300";
                              icon = <HelpCircle size={13} className="text-amber-600 shrink-0" />;
                            } else if (c.evidence_status === "missing" || c.evidence_status === "contradicted") {
                              badgeStyle = "bg-rose-50 text-rose-800 border-rose-300";
                              icon = <AlertTriangle size={13} className="text-rose-600 shrink-0" />;
                            }

                            return (
                              <div
                                key={c.criterion_id}
                                className="p-2.5 bg-white border border-slate-200 rounded-xl space-y-1"
                              >
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex items-start gap-1.5">
                                    {icon}
                                    <span className="text-xs font-bold text-slate-800 leading-snug">
                                      {c.description}
                                    </span>
                                  </div>
                                  <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${badgeStyle} shrink-0`}>
                                    {c.awarded_marks} / {c.max_marks}
                                  </span>
                                </div>

                                {c.evidence_text && (
                                  <p className="text-[11px] text-slate-500 italic pl-5">
                                    "{c.evidence_text}"
                                  </p>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Teacher Override Inline Card */}
                    {!isFinalized ? (
                      <div className="pt-2 border-t border-slate-200/80 space-y-2">
                        <div className="flex items-center justify-between text-[11px] font-bold text-slate-900">
                          <span className="flex items-center gap-1.5">
                            <PenTool size={13} className="text-indigo-600" /> Teacher Score Adjustment
                          </span>
                          <span className="text-slate-400 font-normal text-[10px]">
                            Max: {q.max_marks} pts
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <input
                            type="number"
                            step="0.25"
                            min="0"
                            max={q.max_marks}
                            value={teacherMarksInput}
                            onChange={(e) => setTeacherMarksInput(e.target.value)}
                            placeholder="Score"
                            className="w-24 px-2.5 py-1 bg-white border border-slate-300 rounded-lg text-xs font-bold text-slate-900 focus:outline-hidden focus:border-indigo-500"
                          />
                          <input
                            type="text"
                            value={reasonInput}
                            onChange={(e) => setReasonInput(e.target.value)}
                            placeholder="Reason for change..."
                            className="flex-1 px-2.5 py-1 bg-white border border-slate-300 rounded-lg text-xs text-slate-800 focus:outline-hidden focus:border-indigo-500"
                          />
                          <button
                            onClick={() => handleSaveOverride(q)}
                            disabled={isSubmitting}
                            className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-bold transition-all shrink-0 flex items-center gap-1 cursor-pointer disabled:opacity-50"
                          >
                            <Save size={12} /> {isSubmitting ? "Saving..." : "Save"}
                          </button>
                        </div>

                        {errorMsg && (
                          <div className="text-[11px] text-rose-600 font-semibold bg-rose-50 p-2 rounded-lg border border-rose-200">
                            {errorMsg}
                          </div>
                        )}

                        {successMsg && (
                          <div className="text-[11px] text-emerald-700 font-semibold bg-emerald-50 p-2 rounded-lg border border-emerald-200">
                            {successMsg}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-[11px] text-slate-500 italic bg-slate-100 p-2 rounded-lg flex items-center gap-1 font-semibold">
                        <ShieldCheck size={13} className="text-emerald-600" /> Assessment Finalized — Score locked.
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
