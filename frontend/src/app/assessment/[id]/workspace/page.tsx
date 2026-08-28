"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import {
  AlertTriangle,
  ArrowLeft,
  BarChart2,
  Sparkles,
  HelpCircle,
  ShieldCheck,
  History,
  FileCheck,
  Lock,
  BarChart3,
  ArrowUpRight,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";
import ExtractedQuestionsPanel from "@/components/ExtractedQuestionsPanel";
import OverviewDrawer from "@/components/OverviewDrawer";
import InsightsDrawer from "@/components/InsightsDrawer";
import TeacherReviewQueueModal from "@/components/TeacherReviewQueueModal";
import AuditTrailDrawer from "@/components/AuditTrailDrawer";
import RevisionHistoryDrawer from "@/components/RevisionHistoryDrawer";
import FinalizeAssessmentModal from "@/components/FinalizeAssessmentModal";
import AssessmentAnalyticsPanel from "@/components/AssessmentAnalyticsPanel";

import { getResult, getStructuredResult, fileUrl } from "@/lib/api";
import {
  AssessmentResult,
  StructuredAssessmentResult,
  StructuredQuestionResult,
  UnmatchedAnswer,
  Region,
} from "@/types/assessment";

const AnswerSheetViewer = dynamic(() => import("@/components/AnswerSheetViewer"), { ssr: false });

export default function WorkspacePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();

  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [structuredResult, setStructuredResult] = useState<StructuredAssessmentResult | null>(null);
  const [selectedQuestionId, setSelectedQuestionId] = useState<string | null>(null);
  const [selectedUnmatched, setSelectedUnmatched] = useState<UnmatchedAnswer | null>(null);
  const [activePage, setActivePage] = useState<number>(1);
  const [error, setError] = useState<string | null>(null);

  // Modal / Drawer visibility states
  const [isOverviewOpen, setIsOverviewOpen] = useState(false);
  const [isInsightsOpen, setIsInsightsOpen] = useState(false);
  const [isReviewQueueOpen, setIsReviewQueueOpen] = useState(false);
  const [isAuditTrailOpen, setIsAuditTrailOpen] = useState(false);
  const [isRevisionHistoryOpen, setIsRevisionHistoryOpen] = useState(false);
  const [isFinalizeModalOpen, setIsFinalizeModalOpen] = useState(false);
  const [isAnalyticsOpen, setIsAnalyticsOpen] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Load backend assessment result & structured result on mount
  useEffect(() => {
    if (!params.id) return;
    Promise.all([getResult(params.id), getStructuredResult(params.id)])
      .then(([res, sRes]) => {
        setResult(res);
        setStructuredResult(sRes);
        if (sRes.question_results.length > 0) {
          setSelectedQuestionId(sRes.question_results[0].question_id);
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load workspace data");
      });
  }, [params.id]);

  // Selected Question object
  const selectedQuestion: StructuredQuestionResult | undefined = useMemo(() => {
    if (!structuredResult) return undefined;
    return structuredResult.question_results.find(
      (q) => q.question_id === selectedQuestionId || q.question_number === selectedQuestionId
    );
  }, [structuredResult, selectedQuestionId]);

  // Sync active page whenever selectedQuestion changes
  useEffect(() => {
    if (selectedQuestion) {
      setSelectedUnmatched(null);
      if (selectedQuestion.status === "unanswered") {
        return; // Don't auto-navigate for unanswered
      }
      if (selectedQuestion.answer_pages && selectedQuestion.answer_pages.length > 0) {
        setActivePage(selectedQuestion.answer_pages[0]);
      } else if (selectedQuestion.answer_regions && selectedQuestion.answer_regions.length > 0) {
        const firstP = selectedQuestion.answer_regions[0].page || 1;
        setActivePage(firstP);
      }
    }
  }, [selectedQuestionId, selectedQuestion]);

  // Handle clicking an unmatched region
  const handleSelectUnmatched = (unm: UnmatchedAnswer) => {
    setSelectedUnmatched(unm);
    if (unm.regions && unm.regions.length > 0) {
      setActivePage(unm.regions[0].page || 1);
    }
  };

  // Regions to pass to AnswerSheetViewer
  const viewerRegions: Region[] = useMemo(() => {
    if (selectedUnmatched) {
      return (selectedUnmatched.regions || []).map((r) => ({
        page: r.page || 1,
        bbox: {
          x: r.bbox?.x || 0,
          y: r.bbox?.y || 0,
          width: r.bbox?.width || 0,
          height: r.bbox?.height || 0,
        },
      }));
    }

    if (!selectedQuestion || selectedQuestion.status === "unanswered") {
      return []; // Unanswered questions MUST pass zero regions
    }

    return (selectedQuestion.answer_regions || []).map((r: any) => ({
      page: r.page || 1,
      bbox: {
        x: r.bbox?.x || 0,
        y: r.bbox?.y || 0,
        width: r.bbox?.width || 0,
        height: r.bbox?.height || 0,
      },
    }));
  }, [selectedQuestion, selectedUnmatched]);

  // Filtered review questions list for TeacherReviewQueueModal
  const reviewQuestions = useMemo(() => {
    if (!structuredResult) return [];
    return structuredResult.question_results.filter(
      (q) => q.needs_review || q.review_status === "PENDING_REVIEW"
    );
  }, [structuredResult]);

  // Callback when a question override or finalization updates structured result
  const handleStructuredResultUpdated = (updated: StructuredAssessmentResult) => {
    setStructuredResult(updated);
  };

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f4f5f8] p-4 text-center">
        <div className="bg-white rounded-3xl p-8 max-w-md border border-slate-200 shadow-xl space-y-4">
          <AlertTriangle className="mx-auto text-rose-500" size={36} />
          <h2 className="text-lg font-bold text-slate-800">Workspace Error</h2>
          <p className="text-xs text-slate-500">{error}</p>
          <button
            onClick={() => router.push("/")}
            className="px-5 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors cursor-pointer"
          >
            Return to Upload
          </button>
        </div>
      </div>
    );
  }

  if (!result || !structuredResult) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f4f5f8]">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 border-4 border-slate-900 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs font-semibold text-slate-600">
            Loading Assessment Workspace...
          </p>
        </div>
      </div>
    );
  }

  const isFinalized = structuredResult.assessment_status === "FINALIZED";
  const answerSheetFileUrl = fileUrl(params.id, "answer_sheet");

  return (
    <div className="flex h-screen bg-[#f8fafc] overflow-hidden">
      {/* Left Navigation Sidebar */}
      <Sidebar
        isMobileOpen={isMobileSidebarOpen}
        onMobileClose={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Compact Header Navigation Bar */}
        <header className="h-14 bg-white border-b border-slate-200 px-4 flex items-center justify-between shrink-0 shadow-2xs z-20">
          {/* Left Title & Status */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.push("/")}
              className="p-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors cursor-pointer flex items-center gap-1 text-xs font-semibold"
              title="Return to Upload Dashboard"
            >
              <ArrowLeft size={14} /> Back
            </button>

            <div className="h-4 w-px bg-slate-200" />

            <div className="flex items-center gap-2">
              <span className="font-black text-sm text-slate-900 tracking-tight">
                VedaAI Assessment Workspace
              </span>
              <span
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold ${
                  isFinalized
                    ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                    : "bg-amber-100 text-amber-800 border border-amber-300"
                }`}
              >
                {isFinalized ? (
                  <>
                    <Lock size={11} /> FINALIZED (Rev #{structuredResult.revision_index})
                  </>
                ) : (
                  <>
                    <FileCheck size={11} /> IN REVIEW
                  </>
                )}
              </span>
            </div>
          </div>

          {/* Center Summary Score Badge */}
          <div className="hidden sm:flex items-center gap-2 bg-slate-900 text-white px-3 py-1 rounded-xl shadow-2xs text-xs font-bold">
            <span>Score:</span>
            <span className="text-emerald-400 font-extrabold">{structuredResult.final_awarded_marks}</span>
            <span className="text-slate-400 font-normal">/ {structuredResult.total_max_marks}</span>
            <span className="text-slate-400">({structuredResult.percentage}%)</span>
          </div>

          {/* Right Action Controls for Secondary Teacher Tools */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              onClick={() => setIsOverviewOpen(true)}
              className="px-2.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
              title="View Assessment Overview Metrics"
            >
              <BarChart2 size={13} className="text-slate-600" />
              <span>Overview</span>
            </button>

            <button
              onClick={() => setIsReviewQueueOpen(true)}
              className={`px-2.5 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
                structuredResult.questions_needing_review > 0
                  ? "bg-amber-500 text-white hover:bg-amber-600 shadow-2xs animate-pulse"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              <HelpCircle size={13} />
              <span>Review ({structuredResult.questions_needing_review})</span>
            </button>

            <button
              onClick={() => setIsInsightsOpen(true)}
              className="px-2.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Sparkles size={13} className="text-amber-500" />
              <span>Insights</span>
            </button>

            <button
              onClick={() => setIsAnalyticsOpen(true)}
              className="px-2.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <BarChart3 size={13} className="text-slate-600" />
              <span>Analytics</span>
            </button>

            <button
              onClick={() => setIsAuditTrailOpen(true)}
              className="px-2.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <ShieldCheck size={13} className="text-slate-600" />
              <span>Audit</span>
            </button>

            <button
              onClick={() => setIsRevisionHistoryOpen(true)}
              className="px-2.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <History size={13} className="text-slate-600" />
              <span>Revisions</span>
            </button>

            {!isFinalized ? (
              <button
                onClick={() => setIsFinalizeModalOpen(true)}
                className="px-3 py-1.5 rounded-xl bg-emerald-600 text-white hover:bg-emerald-700 text-xs font-bold flex items-center gap-1.5 shadow-2xs transition-all cursor-pointer"
              >
                <FileCheck size={13} />
                <span>Finalize</span>
              </button>
            ) : (
              <button
                onClick={() => setIsFinalizeModalOpen(true)}
                className="px-3 py-1.5 rounded-xl bg-slate-800 text-slate-200 hover:bg-slate-900 text-xs font-bold flex items-center gap-1.5 shadow-2xs transition-all cursor-pointer"
              >
                <ArrowUpRight size={13} />
                <span>Finalized</span>
              </button>
            )}
          </div>
        </header>

        {/* PRIMARY ASSIGNMENT WORKSPACE (Two-Column Layout) */}
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
          {/* LEFT COLUMN: Extracted Questions List (~40% width) */}
          <div className="w-full md:w-[420px] lg:w-[460px] xl:w-[500px] h-full shrink-0 flex flex-col overflow-hidden">
            <ExtractedQuestionsPanel
              assessmentId={params.id}
              structuredResult={structuredResult}
              unmatchedAnswers={result.unmatched_answers || []}
              selectedId={selectedQuestionId}
              onSelectQuestion={setSelectedQuestionId}
              onSelectUnmatched={handleSelectUnmatched}
              onStructuredResultUpdated={handleStructuredResultUpdated}
            />
          </div>

          {/* RIGHT COLUMN: Student Answer Sheet Viewer (~60% width) */}
          <div className="flex-1 h-full min-h-[400px] bg-[#eef0f4] flex flex-col overflow-hidden">
            <AnswerSheetViewer
              fileUrl={answerSheetFileUrl}
              isPdf={result.answer_sheet_is_pdf}
              questionNumber={
                selectedUnmatched
                  ? "Unmatched"
                  : selectedQuestion?.question_number || ""
              }
              regions={viewerRegions}
              pageSizes={result.answer_sheet_page_sizes}
              activePage={activePage}
              onPageChange={setActivePage}
              totalPages={result.answer_sheet_pages}
            />
          </div>
        </div>
      </div>

      {/* Drawers & Modals for Secondary Teacher Tools */}
      <OverviewDrawer
        isOpen={isOverviewOpen}
        onClose={() => setIsOverviewOpen(false)}
        structuredResult={structuredResult}
        onOpenReviewQueue={() => setIsReviewQueueOpen(true)}
        onOpenFinalize={() => setIsFinalizeModalOpen(true)}
        onOpenAuditTrail={() => setIsAuditTrailOpen(true)}
        onOpenRevisionHistory={() => setIsRevisionHistoryOpen(true)}
        onOpenAnalytics={() => setIsAnalyticsOpen(true)}
      />

      <InsightsDrawer
        isOpen={isInsightsOpen}
        onClose={() => setIsInsightsOpen(false)}
        assessmentId={params.id}
        onSelectQuestion={(qId) => {
          setSelectedQuestionId(qId);
          setIsInsightsOpen(false);
        }}
      />

      <TeacherReviewQueueModal
        isOpen={isReviewQueueOpen}
        onClose={() => setIsReviewQueueOpen(false)}
        reviewQuestions={reviewQuestions}
        onSelectQuestion={setSelectedQuestionId}
      />

      <AuditTrailDrawer
        isOpen={isAuditTrailOpen}
        onClose={() => setIsAuditTrailOpen(false)}
        auditEvents={structuredResult.audit_trail || []}
      />

      <RevisionHistoryDrawer
        isOpen={isRevisionHistoryOpen}
        onClose={() => setIsRevisionHistoryOpen(false)}
        assessmentId={params.id}
        revisions={structuredResult.version_history || []}
      />

      <FinalizeAssessmentModal
        isOpen={isFinalizeModalOpen}
        onClose={() => setIsFinalizeModalOpen(false)}
        structuredResult={structuredResult}
        onFinalized={handleStructuredResultUpdated}
      />

      <AssessmentAnalyticsPanel
        isOpen={isAnalyticsOpen}
        onClose={() => setIsAnalyticsOpen(false)}
        structuredResult={structuredResult}
      />
    </div>
  );
}
