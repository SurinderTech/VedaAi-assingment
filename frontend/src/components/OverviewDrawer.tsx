"use client";

import { StructuredAssessmentResult } from "@/types/assessment";
import { X, BarChart2 } from "lucide-react";
import TeacherDashboardOverview from "./TeacherDashboardOverview";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  structuredResult: StructuredAssessmentResult;
  onOpenReviewQueue: () => void;
  onOpenFinalize: () => void;
  onOpenAuditTrail: () => void;
  onOpenRevisionHistory: () => void;
  onOpenAnalytics: () => void;
}

export default function OverviewDrawer({
  isOpen,
  onClose,
  structuredResult,
  onOpenReviewQueue,
  onOpenFinalize,
  onOpenAuditTrail,
  onOpenRevisionHistory,
  onOpenAnalytics,
}: Props) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-50 flex justify-end">
      <div className="bg-white max-w-4xl w-full h-full shadow-2xl flex flex-col border-l border-slate-200 overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50 shrink-0">
          <div className="flex items-center gap-2">
            <BarChart2 size={18} className="text-indigo-600" />
            <h2 className="text-sm font-bold text-slate-900">Assessment Summary & Metrics Overview</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <TeacherDashboardOverview
            structuredResult={structuredResult}
            onOpenReviewQueue={() => {
              onClose();
              onOpenReviewQueue();
            }}
            onOpenFinalize={() => {
              onClose();
              onOpenFinalize();
            }}
            onOpenAuditTrail={() => {
              onClose();
              onOpenAuditTrail();
            }}
            onOpenRevisionHistory={() => {
              onClose();
              onOpenRevisionHistory();
            }}
            onOpenAnalytics={() => {
              onClose();
              onOpenAnalytics();
            }}
          />
        </div>
      </div>
    </div>
  );
}
