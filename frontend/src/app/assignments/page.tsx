"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { FileText, Plus, ArrowRight, CheckCircle2 } from "lucide-react";
import { useRouter } from "next/navigation";

export default function AssignmentsPage() {
  const router = useRouter();
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const ASSIGNMENTS = [
    { title: "Mid-Term Physics Question Paper & Sheet", class: "Class 10-A", due: "Today", status: "Active Mapping" },
    { title: "Calculus Subquestions & Diagrams Evaluation", class: "Class 11-A", due: "Yesterday", status: "Completed" },
  ];

  return (
    <div className="min-h-screen w-full bg-[#f4f5f8] flex flex-col lg:flex-row font-sans">
      <Sidebar isMobileOpen={isMobileOpen} onMobileClose={() => setIsMobileOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0 min-h-screen bg-gradient-to-b from-white via-[#f7f8fa] to-[#e4e7ed]">
        <Header title="Assignments" onMenuClick={() => setIsMobileOpen(true)} />

        <main className="flex-1 p-4 sm:p-8 max-w-5xl mx-auto w-full space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200/80 pb-5">
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900">Assignments &amp; Exams</h1>
              <p className="text-xs text-slate-500 mt-1">Manage assessment uploads and evaluation workflows</p>
            </div>

            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#ff5a1f] hover:bg-[#e04810] text-white text-xs font-bold transition-all shadow-sm cursor-pointer self-start sm:self-auto"
            >
              <Plus size={16} />
              <span>Create New Assignment</span>
            </button>
          </div>

          <div className="space-y-3">
            {ASSIGNMENTS.map((a, i) => (
              <div
                key={i}
                className="bg-white rounded-3xl p-5 border border-slate-200 shadow-2xs hover:shadow-md transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4"
              >
                <div className="flex items-start gap-3.5">
                  <div className="h-10 w-10 rounded-2xl bg-[#fff0ea] text-[#ff5a1f] flex items-center justify-center font-bold shrink-0 mt-0.5">
                    <FileText size={20} />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">{a.title}</h3>
                    <div className="flex items-center gap-3 text-xs text-slate-500 mt-1">
                      <span>{a.class}</span>
                      <span>&bull;</span>
                      <span className="flex items-center gap-1 text-emerald-600 font-medium">
                        <CheckCircle2 size={12} /> {a.status}
                      </span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => router.push("/")}
                  className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-[#ff5a1f] text-white text-xs font-bold transition-colors cursor-pointer flex items-center gap-1.5 self-start sm:self-auto"
                >
                  <span>Open Assessment</span>
                  <ArrowRight size={13} />
                </button>
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
