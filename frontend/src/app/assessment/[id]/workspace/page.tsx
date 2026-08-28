"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { AlertTriangle, ArrowLeft } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import TeacherDashboardOverview from "@/components/TeacherDashboardOverview";
import AssessmentInsightsPanel from "@/components/AssessmentInsightsPanel";
import QuestionNavigationPanel from "@/components/QuestionNavigationPanel";
import QuestionReviewWorkspace from "@/components/QuestionReviewWorkspace";
import TeacherReviewQueueModal from "@/components/TeacherReviewQueueModal";
import AuditTrailDrawer from "@/components/AuditTrailDrawer";
import RevisionHistoryDrawer from "@/components/RevisionHistoryDrawer";
import FinalizeAssessmentModal from "@/components/FinalizeAssessmentModal";
import AssessmentAnalyticsPanel from "@/components/AssessmentAnalyticsPanel";

import { getResult, getStructuredResult } from "@/lib/api";
import {
  AssessmentResult,
  StructuredAssessmentResult,
  StructuredQuestionResult,
} from "@/types/assessment";

export default function WorkspacePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();

  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [structuredResult, setStructuredResult] = useState<StructuredAssessmentResult | null>(null);
  const [selectedQuestionId, setSelectedQuestionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Modal / Drawer visibility states
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

  // Filtered review questions list
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
            className="px-5 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors"
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
            Loading Step 6 Intelligent Teacher Workspace...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#f4f5f8] overflow-hidden">
      {/* Left Navigation Sidebar */}
      <Sidebar
        isMobileOpen={isMobileSidebarOpen}
        onMobileClose={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Workspace Layout */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        <Header onMenuClick={() => setIsMobileSidebarOpen(true)} />

        {/* Assessment Overview Header */}
        <TeacherDashboardOverview
          structuredResult={structuredResult}
          onOpenReviewQueue={() => setIsReviewQueueOpen(true)}
          onOpenFinalize={() => setIsFinalizeModalOpen(true)}
          onOpenAuditTrail={() => setIsAuditTrailOpen(true)}
          onOpenRevisionHistory={() => setIsRevisionHistoryOpen(true)}
          onOpenAnalytics={() => setIsAnalyticsOpen(true)}
        />

        {/* Step 9 Assessment Insights Panel */}
        <div className="px-4 sm:px-6 pt-3 shrink-0">
          <AssessmentInsightsPanel
            assessmentId={params.id}
            onSelectQuestion={(qId) => setSelectedQuestionId(qId)}
          />
        </div>

        {/* Interactive Workspace Body */}
        <div className="flex-1 flex overflow-hidden">
          {/* Question Navigation Sidebar */}
          <QuestionNavigationPanel
            questions={structuredResult.question_results}
            selectedId={selectedQuestionId}
            onSelectQuestion={setSelectedQuestionId}
          />

          {/* Question Detail & BBox Viewer & AI Evidence Panel */}
          {selectedQuestion ? (
            <QuestionReviewWorkspace
              assessmentId={params.id}
              question={selectedQuestion}
              structuredResult={structuredResult}
              answerSheetPages={result.answer_sheet_pages}
              answerSheetPageSizes={result.answer_sheet_page_sizes}
              answerSheetIsPdf={result.answer_sheet_is_pdf}
              onQuestionUpdated={handleStructuredResultUpdated}
            />
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-400 text-xs font-semibold">
              Select a question to inspect answer sheet and AI evidence
            </div>
          )}
        </div>
      </div>

      {/* Modals & Drawers */}
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
