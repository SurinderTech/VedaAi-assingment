"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { Users, GraduationCap, ArrowRight, Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";

export default function ClassroomPage() {
  const router = useRouter();
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const CLASSES = [
    { name: "Class 10-A", section: "Mathematics & Science", students: 32, status: "Active Evaluation" },
    { name: "Class 10-B", section: "Physics & Chemistry", students: 28, status: "Pending Upload" },
    { name: "Class 11-A", section: "Advanced Calculus", students: 35, status: "Completed" },
  ];

  return (
    <div className="min-h-screen w-full bg-[#f4f5f8] flex flex-col lg:flex-row font-sans">
      <Sidebar isMobileOpen={isMobileOpen} onMobileClose={() => setIsMobileOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0 min-h-screen bg-gradient-to-b from-white via-[#f7f8fa] to-[#e4e7ed]">
        <Header title="My Classroom" onMenuClick={() => setIsMobileOpen(true)} />

        <main className="flex-1 p-4 sm:p-8 max-w-5xl mx-auto w-full space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200/80 pb-5">
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900">Classroom Roster</h1>
              <p className="text-xs text-slate-500 mt-1">Delhi Public School — Examiner Classroom Workspace</p>
            </div>

            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#1a1a1a] hover:bg-[#ff5a1f] text-white text-xs font-bold transition-all shadow-sm cursor-pointer self-start sm:self-auto"
            >
              <span>Evaluate New Assessment</span>
              <ArrowRight size={14} />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
            {CLASSES.map((c) => (
              <div
                key={c.name}
                className="bg-white rounded-3xl p-5 border border-slate-200 shadow-2xs hover:shadow-md transition-all space-y-4"
              >
                <div className="flex items-center justify-between">
                  <div className="h-10 w-10 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center font-bold">
                    <GraduationCap size={20} />
                  </div>
                  <span className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 text-[10px] font-bold">
                    {c.students} Students
                  </span>
                </div>

                <div>
                  <h3 className="text-base font-bold text-slate-900">{c.name}</h3>
                  <p className="text-xs text-slate-500 mt-0.5">{c.section}</p>
                </div>

                <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs font-semibold text-slate-700">
                  <span className="text-[#ff5a1f] flex items-center gap-1">
                    <Sparkles size={13} /> {c.status}
                  </span>
                  <button
                    onClick={() => router.push("/")}
                    className="hover:underline text-slate-900 font-bold cursor-pointer"
                  >
                    Open &rarr;
                  </button>
                </div>
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
