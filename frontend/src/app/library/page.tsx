"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { Bookmark, FileText, ArrowRight, Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";
import { listAssessments } from "@/lib/api";

export default function LibraryPage() {
  const router = useRouter();
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [items, setItems] = useState<any[]>([]);

  useEffect(() => {
    listAssessments().then((res) => setItems(res));
  }, []);

  return (
    <div className="min-h-screen w-full bg-[#f4f5f8] flex flex-col lg:flex-row font-sans">
      <Sidebar isMobileOpen={isMobileOpen} onMobileClose={() => setIsMobileOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0 min-h-screen bg-gradient-to-b from-white via-[#f7f8fa] to-[#e4e7ed]">
        <Header title="My Library" onMenuClick={() => setIsMobileOpen(true)} />

        <main className="flex-1 p-4 sm:p-8 max-w-5xl mx-auto w-full space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200/80 pb-5">
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900">Assessment Document Library</h1>
              <p className="text-xs text-slate-500 mt-1">Processed question papers, handwritten answer sheets, and OCR result stores</p>
            </div>

            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#1a1a1a] hover:bg-[#ff5a1f] text-white text-xs font-bold transition-all shadow-sm cursor-pointer self-start sm:self-auto"
            >
              <span>Upload New Assessment</span>
              <ArrowRight size={14} />
            </button>
          </div>

          {items.length === 0 ? (
            <div className="bg-white rounded-3xl p-12 border border-slate-200 text-center space-y-4 shadow-2xs">
              <div className="mx-auto h-12 w-12 rounded-2xl bg-[#fff0ea] text-[#ff5a1f] flex items-center justify-center">
                <Bookmark size={24} />
              </div>
              <h3 className="text-base font-bold text-slate-900">No stored assessments yet</h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Uploaded question papers and answer sheets will appear in your in-memory assessment library once evaluated.
              </p>
              <button
                onClick={() => router.push("/")}
                className="px-6 py-3 rounded-full bg-[#ff5a1f] text-white text-xs font-bold shadow-md hover:bg-[#e04810] transition-colors cursor-pointer"
              >
                Create Assessment
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {items.map((item) => (
                <div
                  key={item.assessment_id}
                  className="bg-white rounded-3xl p-5 border border-slate-200 shadow-2xs hover:shadow-md transition-all space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-slate-400">ID: {item.assessment_id}</span>
                    <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 text-[10px] font-bold uppercase">
                      {item.state}
                    </span>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-xl bg-[#fff0ea] text-[#ff5a1f] flex items-center justify-center shrink-0">
                      <FileText size={20} />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-slate-900">{item.question_count} Questions Extracted</h4>
                      <p className="text-xs text-slate-500">
                        {item.matched_count} Matched &bull; {item.unanswered_count} Unanswered
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={() => router.push(`/assessment/${item.assessment_id}/workspace`)}
                    className="w-full py-2.5 rounded-xl bg-slate-100 hover:bg-[#ff5a1f] hover:text-white text-slate-700 text-xs font-bold transition-colors cursor-pointer flex items-center justify-center gap-1"
                  >
                    <span>Open Workspace</span>
                    <ArrowRight size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
