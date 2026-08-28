"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { AlertTriangle, ArrowLeft } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import AssessmentReportViewer from "@/components/AssessmentReportViewer";
import { getStudentReport } from "@/lib/api";
import { StudentAssessmentReport } from "@/types/assessment";

export default function StudentResultsPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();

  const [report, setReport] = useState<StudentAssessmentReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  useEffect(() => {
    if (!params.id) return;
    getStudentReport(params.id)
      .then((rep) => setReport(rep))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load report"));
  }, [params.id]);

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f4f5f8] p-4 text-center">
        <div className="bg-white rounded-3xl p-8 max-w-md border border-slate-200 shadow-xl space-y-4">
          <AlertTriangle className="mx-auto text-rose-500" size={36} />
          <h2 className="text-lg font-bold text-slate-800">Report Error</h2>
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

  if (!report) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f4f5f8]">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 border-4 border-slate-900 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs font-semibold text-slate-600">
            Generating Evidence-Grounded Student Report...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#f4f5f8] overflow-hidden">
      <Sidebar
        isMobileOpen={isMobileSidebarOpen}
        onMobileClose={() => setIsMobileSidebarOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0 h-full overflow-y-auto">
        <Header onMenuClick={() => setIsMobileSidebarOpen(true)} />
        <AssessmentReportViewer report={report} />
      </div>
    </div>
  );
}
