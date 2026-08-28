"use client";

import { useEffect, useState } from "react";
import { StructuredQuestionResult, StructuredAssessmentResult } from "@/types/assessment";
import { overrideQuestionMarks, fileUrl } from "@/lib/api";
import dynamic from "next/dynamic";
const AnswerSheetViewer = dynamic(() => import("@/components/AnswerSheetViewer"), { ssr: false });
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Circle,
  PenTool,
  Sparkles,
  ShieldCheck,
  FileText,
  Save,
  RotateCcw,
  BookOpen,
} from "lucide-react";

interface Props {
  assessmentId: string;
  question: StructuredQuestionResult;
  structuredResult: StructuredAssessmentResult;
  answerSheetPages: number;
  answerSheetPageSizes: number[][];
  answerSheetIsPdf: boolean;
  onQuestionUpdated: (updated: StructuredAssessmentResult) => void;
}

export default function QuestionReviewWorkspace({
  assessmentId,
  question,
  structuredResult,
  answerSheetPages,
  answerSheetPageSizes,
  answerSheetIsPdf,
  onQuestionUpdated,
}: Props) {
  const [activePage, setActivePage] = useState<number>(1);
  const [teacherMarksInput, setTeacherMarksInput] = useState<string>(
    question.teacher_adjusted_marks !== null
      ? String(question.teacher_adjusted_marks)
      : String(question.awarded_marks)
  );
  const [criterionInputs, setCriterionInputs] = useState<Record<string, string>>({});
  const [reasonInput, setReasonInput] = useState("");
  const [commentInput, setCommentInput] = useState("");
  const [feedbackInput, setFeedbackInput] = useState(question.feedback || "");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const isFinalized = structuredResult.assessment_status === "FINALIZED";

  // Synchronize inputs when selected question changes
  useEffect(() => {
    setTeacherMarksInput(
      question.teacher_adjusted_marks !== null
        ? String(question.teacher_adjusted_marks)
        : String(question.awarded_marks)
    );
    const initialCr: Record<string, string> = {};
    question.criterion_results.forEach((c) => {
      initialCr[c.criterion_id] = String(c.awarded_marks);
    });
    setCriterionInputs(initialCr);
    setFeedbackInput(question.feedback || "");
    setErrorMsg(null);
    setSuccessMsg(null);

    if (question.answer_pages && question.answer_pages.length > 0) {
      setActivePage(question.answer_pages[0]);
    }
  }, [question]);

  const answerSheetFileUrl = fileUrl(assessmentId, "answer_sheet");

  // Format region objects for AnswerSheetViewer
  const viewerRegions = (question.answer_regions || []).map((r) => ({
    page: r.page || 1,
    bbox: {
      x: r.bbox?.x || 0,
      y: r.bbox?.y || 0,
      width: r.bbox?.width || 0,
      height: r.bbox?.height || 0,
    },
  }));

  const handleCriterionChange = (criterionId: string, val: string) => {
    const updated = { ...criterionInputs, [criterionId]: val };
    setCriterionInputs(updated);

    // Auto-sum criterion marks for question override prediction
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

  const handleSaveOverride = async () => {
    setIsSubmitting(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    const parsedScore = parseFloat(teacherMarksInput);
    if (isNaN(parsedScore) || parsedScore < 0 || parsedScore > question.max_marks) {
      setErrorMsg(`Teacher marks must be a valid number between 0.0 and ${question.max_marks}`);
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
        reviewer: "Dr. Smith",
      });
      setSuccessMsg("Teacher override saved successfully!");
      onQuestionUpdated(updated);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Override failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col md:flex-row h-full overflow-hidden bg-[#f4f5f8]">
      {/* COLUMN 1: Question Details & Student Answer */}
      <div className="w-full md:w-80 lg:w-96 bg-white border-r border-slate-200 flex flex-col overflow-y-auto shrink-0 p-4 space-y-4">
        {/* Question Header Card */}
        <div className="bg-slate-50 p-3.5 rounded-2xl border border-slate-200/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-black text-sm text-slate-900 bg-white border border-slate-200 px-2.5 py-0.5 rounded-lg">
              Q{question.question_number}
            </span>
            <span className="text-xs font-bold text-slate-700">
              Max Marks: <span className="text-indigo-600">{question.max_marks} pts</span>
            </span>
          </div>
          <p className="text-xs text-slate-800 font-medium leading-relaxed">
            {question.question_text || "Question prompt text"}
          </p>
        </div>

        {/* Student OCR Answer Card */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-slate-800">
            <span className="flex items-center gap-1.5">
              <FileText size={14} className="text-slate-600" /> Extracted Student Answer
            </span>
            {question.answer_pages && question.answer_pages.length > 0 && (
              <span className="text-[10px] font-mono bg-slate-100 border border-slate-200 px-2 py-0.5 rounded-full text-slate-600">
                Page {question.answer_pages.join(", ")}
              </span>
            )}
          </div>

          <div className="bg-slate-900 text-slate-100 p-3.5 rounded-2xl font-mono text-xs leading-relaxed max-h-56 overflow-y-auto whitespace-pre-wrap">
            {question.answer_text ? (
              question.answer_text
            ) : (
              <span className="text-slate-500 italic">No answer text extracted (Unanswered)</span>
            )}
          </div>

          {/* Spanned Pages Tag */}
          {question.answer_pages && question.answer_pages.length > 1 && (
            <div className="bg-amber-50 border border-amber-200 p-2 rounded-xl text-[11px] text-amber-800 flex items-center justify-between font-semibold">
              <span>Spans Multi-Pages:</span>
              <div className="flex items-center gap-1">
                {question.answer_pages.map((p) => (
                  <button
                    key={p}
                    onClick={() => setActivePage(p)}
                    className={`px-2 py-0.5 rounded-md font-bold transition-all ${
                      p === activePage ? "bg-amber-600 text-white shadow-xs" : "bg-amber-100 text-amber-900 hover:bg-amber-200"
                    }`}
                  >
                    Page {p}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Evidence Summary List */}
        {question.evidence_summary && question.evidence_summary.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
              <Sparkles size={14} className="text-amber-500" /> Evidence Key Points
            </div>
            <div className="space-y-1 text-xs">
              {question.evidence_summary.map((ev, idx) => (
                <div key={idx} className="p-2 bg-slate-50 border border-slate-200/60 rounded-xl text-slate-700">
                  • {ev}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* COLUMN 2: Original Answer-Sheet Viewer (Middle) */}
      <div className="flex-1 h-full min-h-[400px] border-r border-slate-200 bg-[#eef0f4] flex flex-col">
        <AnswerSheetViewer
          fileUrl={answerSheetFileUrl}
          isPdf={answerSheetIsPdf}
          questionNumber={question.question_number}
          regions={viewerRegions}
          pageSizes={answerSheetPageSizes}
          activePage={activePage}
          onPageChange={setActivePage}
          totalPages={answerSheetPages}
        />
      </div>

      {/* COLUMN 3: AI Evidence & Teacher Override Panel (Right) */}
      <div className="w-full md:w-96 lg:w-[420px] bg-white flex flex-col overflow-y-auto shrink-0 p-4 space-y-5">
        {/* AI Evaluation Meta Header */}
        <div className="bg-slate-50 p-4 rounded-2xl border border-slate-200/80 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Sparkles size={16} className="text-indigo-600" />
              <span className="font-bold text-xs text-slate-900">AI Evaluation Analysis</span>
            </div>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-slate-200 text-slate-700">
              {question.grading_provenance || "local"}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-white p-2.5 rounded-xl border border-slate-200/70">
              <div className="text-[10px] text-slate-400 font-semibold">Original AI Marks</div>
              <div className="text-lg font-black text-slate-900">
                {question.original_ai_marks} <span className="text-xs font-normal text-slate-500">/ {question.max_marks}</span>
              </div>
            </div>

            <div className="bg-white p-2.5 rounded-xl border border-slate-200/70">
              <div className="text-[10px] text-slate-400 font-semibold">Confidence</div>
              <div className="text-lg font-black text-emerald-600">
                {Math.round(question.evaluation_confidence * 100)}%
              </div>
            </div>
          </div>

          {question.escalation_reason && (
            <div className="bg-amber-50 border border-amber-200 p-2 rounded-xl text-[11px] text-amber-800 flex items-center gap-1 font-semibold">
              <AlertTriangle size={13} className="shrink-0" />
              <span>Escalation: {question.escalation_reason}</span>
            </div>
          )}
        </div>

        {/* Criterion Breakdown */}
        <div className="space-y-2.5">
          <div className="text-xs font-bold text-slate-900 flex items-center justify-between">
            <span>Rubric Criteria Evaluation ({question.criterion_results.length})</span>
            <span className="text-[10px] font-normal text-slate-400">Step 4 Evidence Fusion</span>
          </div>

          {question.criterion_results.length === 0 ? (
            <div className="p-4 text-center text-xs text-slate-400 bg-slate-50 rounded-xl">
              No individual criteria defined for this question
            </div>
          ) : (
            question.criterion_results.map((c) => {
              let badgeColor = "bg-emerald-100 text-emerald-800 border-emerald-300";
              if (c.evidence_status === "partially_present") {
                badgeColor = "bg-amber-100 text-amber-800 border-amber-300";
              } else if (c.evidence_status === "missing" || c.evidence_status === "contradicted") {
                badgeColor = "bg-rose-100 text-rose-800 border-rose-300";
              }

              return (
                <div key={c.criterion_id} className="p-3 bg-white border border-slate-200 rounded-2xl space-y-2">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-xs font-bold text-slate-800 leading-snug">
                      {c.description}
                    </span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${badgeColor} shrink-0`}>
                      {c.evidence_status}
                    </span>
                  </div>

                  {c.evidence_text && (
                    <div className="bg-slate-50 p-2 rounded-xl text-xs text-slate-600 italic">
                      "{c.evidence_text}"
                    </div>
                  )}

                  <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
                    <span>Source: <span className="font-mono text-slate-700">{c.provenance}</span></span>
                    <div className="flex items-center gap-2">
                      <span>Conf: {Math.round(c.confidence * 100)}%</span>
                      <span className="font-bold text-slate-900">{c.awarded_marks} / {c.max_marks} pts</span>
                    </div>
                  </div>

                  {/* Teacher Criterion Adjustment Input */}
                  {!isFinalized && (
                    <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs">
                      <label className="text-[11px] font-semibold text-slate-600">Adjust Criterion:</label>
                      <input
                        type="number"
                        step="0.25"
                        min="0"
                        max={c.max_marks}
                        value={criterionInputs[c.criterion_id] ?? String(c.awarded_marks)}
                        onChange={(e) => handleCriterionChange(c.criterion_id, e.target.value)}
                        className="w-20 px-2 py-1 border border-slate-200 rounded-lg text-right font-bold text-slate-900 focus:outline-hidden focus:border-indigo-500 text-xs"
                      />
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* AI vs Teacher Comparison View */}
        <div className="bg-slate-50 p-4 rounded-2xl border border-slate-200/80 space-y-2.5">
          <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
            <BookOpen size={14} className="text-indigo-600" /> AI vs Teacher Decision Comparison
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 bg-white rounded-xl border border-slate-200">
              <div className="text-[10px] text-slate-400 font-bold">AI Evaluation</div>
              <div className="text-lg font-black text-slate-900">{question.original_ai_marks} / {question.max_marks}</div>
              <div className="text-[10px] text-slate-500">{Math.round(question.evaluation_confidence * 100)}% confidence</div>
            </div>

            <div className="p-3 bg-indigo-50/60 rounded-xl border border-indigo-200">
              <div className="text-[10px] text-indigo-700 font-bold">Teacher Decision</div>
              <div className="text-lg font-black text-indigo-900">
                {question.teacher_adjusted_marks !== null ? `${question.teacher_adjusted_marks} / ${question.max_marks}` : "Not Overridden"}
              </div>
              <div className="text-[10px] text-indigo-600 font-semibold">
                Status: {question.review_status}
              </div>
            </div>
          </div>
        </div>

        {/* Teacher Override Form */}
        {!isFinalized ? (
          <div className="p-4 bg-white border border-slate-200 rounded-2xl space-y-3 shadow-xs">
            <div className="flex items-center justify-between text-xs font-bold text-slate-900">
              <span className="flex items-center gap-1.5">
                <PenTool size={14} className="text-indigo-600" /> Apply Teacher Override
              </span>
              <span className="text-[10px] text-slate-400 font-normal">Range: 0.0 - {question.max_marks}</span>
            </div>

            <div>
              <label className="block text-[11px] font-bold text-slate-700 mb-1">
                Question Awarded Marks
              </label>
              <input
                type="number"
                step="0.25"
                min="0"
                max={question.max_marks}
                value={teacherMarksInput}
                onChange={(e) => setTeacherMarksInput(e.target.value)}
                className="w-full px-3 py-2 border border-slate-200 rounded-xl text-sm font-extrabold text-slate-900 focus:outline-hidden focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-[11px] font-bold text-slate-700 mb-1">
                Override Reason
              </label>
              <input
                type="text"
                value={reasonInput}
                onChange={(e) => setReasonInput(e.target.value)}
                placeholder="e.g. Minor notation penalty"
                className="w-full px-3 py-1.5 border border-slate-200 rounded-xl text-xs text-slate-800 focus:outline-hidden focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-[11px] font-bold text-slate-700 mb-1">
                Teacher Comment
              </label>
              <textarea
                value={commentInput}
                onChange={(e) => setCommentInput(e.target.value)}
                rows={2}
                placeholder="Teacher feedback for student..."
                className="w-full px-3 py-1.5 border border-slate-200 rounded-xl text-xs text-slate-800 focus:outline-hidden focus:border-indigo-500"
              />
            </div>

            {errorMsg && (
              <div className="p-2.5 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center gap-1.5">
                <AlertTriangle size={14} className="shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {successMsg && (
              <div className="p-2.5 bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs rounded-xl flex items-center gap-1.5 font-bold">
                <CheckCircle2 size={14} className="shrink-0" />
                <span>{successMsg}</span>
              </div>
            )}

            <button
              onClick={handleSaveOverride}
              disabled={isSubmitting}
              className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-bold transition-all shadow-xs flex items-center justify-center gap-1.5"
            >
              <Save size={14} />
              {isSubmitting ? "Saving Override..." : "Submit Teacher Override"}
            </button>
          </div>
        ) : (
          <div className="p-4 bg-slate-100 rounded-2xl border border-slate-200 text-xs text-slate-600 space-y-1">
            <div className="font-bold text-slate-800 flex items-center gap-1">
              <ShieldCheck size={14} className="text-emerald-600" /> Revision Locked
            </div>
            <p className="text-[11px]">
              This assessment revision is finalized. Further overrides will create a new version revision.
            </p>
          </div>
        )}

        {/* Feedback Section */}
        <div className="space-y-1.5 pt-2">
          <div className="text-xs font-bold text-slate-900 flex items-center justify-between">
            <span>Evidence-Grounded Student Feedback</span>
            <span className="text-[10px] text-slate-400">Read-Only Safety</span>
          </div>

          <textarea
            value={feedbackInput}
            onChange={(e) => setFeedbackInput(e.target.value)}
            disabled={isFinalized}
            rows={3}
            className="w-full p-3 bg-slate-50 border border-slate-200 rounded-2xl text-xs text-slate-800 focus:outline-hidden focus:border-slate-400 font-medium"
          />
        </div>
      </div>
    </div>
  );
}
