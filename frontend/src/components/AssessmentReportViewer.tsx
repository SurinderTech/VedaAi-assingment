"use client";

import { StudentAssessmentReport } from "@/types/assessment";
import StudentResultOverview from "@/components/StudentResultOverview";
import StudentFeedbackPanel from "@/components/StudentFeedbackPanel";
import ImprovementRecommendations from "@/components/ImprovementRecommendations";
import PerformanceBreakdown from "@/components/PerformanceBreakdown";
import QuestionPerformanceList from "@/components/QuestionPerformanceList";

interface Props {
  report: StudentAssessmentReport;
}

export default function AssessmentReportViewer({ report }: Props) {
  return (
    <div className="space-y-6">
      <StudentResultOverview summary={report.performance_summary} revisionIndex={report.report_version} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 space-y-6 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <StudentFeedbackPanel report={report} />
            <QuestionPerformanceList questions={report.question_results} />
          </div>

          <div className="space-y-6">
            <PerformanceBreakdown report={report} />
            <ImprovementRecommendations report={report} />
          </div>
        </div>
      </div>
    </div>
  );
}
