"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Check, AlertCircle } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import VedaLogo from "@/components/VedaLogo";
import { getStatus } from "@/lib/api";
import { AssessmentStatus, ProcessingState } from "@/types/assessment";

const STAGES: { state: ProcessingState; label: string; desc: string }[] = [
  { state: "extracting_questions", label: "Reading Question Paper", desc: "Extracting printed questions and sub-parts" },
  { state: "extracting_answers", label: "Scanning Handwritten Sheet", desc: "OCR line segmentation and bounding boxes" },
  { state: "mapping", label: "AI Answer Mapping", desc: "Matching student responses to questions" },
  { state: "completed", label: "Finalizing Workspace", desc: "Preparing interactive evaluation view" },
];

function stageIndex(state: ProcessingState): number {
  const order: ProcessingState[] = [
    "uploaded", "processing", "extracting_questions", "extracting_answers", "mapping", "grading", "completed",
  ];
  return order.indexOf(state);
}

export default function ProcessingPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [status, setStatus] = useState<AssessmentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const s = await getStatus(params.id);
        if (cancelled) return;
        setStatus(s);
        if (s.state === "completed") {
          router.push(`/assessment/${params.id}/workspace`);
          return;
        }
        if (s.state === "failed") {
          setError(s.message || "Processing failed.");
          return;
        }
        timer = setTimeout(poll, 1200);
      } catch {
        if (!cancelled) {
          timer = setTimeout(poll, 1500);
        }
      }
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [params.id, router]);

  const currentIndex = status ? stageIndex(status.state) : 0;

  return (
    <div className="min-h-screen w-full bg-[#f4f5f8] flex flex-col lg:flex-row font-sans">
      <Sidebar isMobileOpen={isMobileOpen} onMobileClose={() => setIsMobileOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0 min-h-screen bg-gradient-to-b from-white via-[#f7f8fa] to-[#e4e7ed]">
        <Header title="Assessment Processing" onMenuClick={() => setIsMobileOpen(true)} />

        <main className="flex-1 flex items-center justify-center px-4 py-8 sm:py-12">
          <div className="w-full max-w-lg bg-white rounded-3xl border border-slate-200/80 p-6 sm:p-10 shadow-xl shadow-slate-900/5 text-center">
            {/* Pulsing Big V Logo */}
            <div className="flex justify-center mb-6">
              <div className="relative">
                <div className="absolute -inset-3 rounded-3xl bg-[#ff5a1f]/20 animate-ping opacity-75" style={{ animationDuration: "3s" }} />
                <VedaLogo size="xl" showText={false} />
              </div>
            </div>

            <div className="mb-2">
              <VedaLogo size="md" className="justify-center mb-2" />
            </div>

            <h1 className="text-xl font-bold text-slate-900 mb-1">
              {error ? "Processing Error" : "Evaluating Assessment"}
            </h1>
            <p className="text-xs text-slate-500 mb-8">
              {error ? error : "AI is segmenting questions, OCR bounding boxes, and student answers"}
            </p>

            {!error && (
              <div className="space-y-3 text-left">
                {STAGES.map((stage) => {
                  const idx = stageIndex(stage.state);
                  const done = currentIndex > idx || status?.state === "completed";
                  const active = status?.state === stage.state;
                  return (
                    <div
                      key={stage.state}
                      className={`flex items-center gap-3.5 rounded-2xl border p-4 transition-all duration-300 ${
                        active
                          ? "border-[#ff5a1f] bg-[#fff0e8]/50 shadow-xs scale-[1.01]"
                          : done
                          ? "border-emerald-200 bg-emerald-50/40"
                          : "border-slate-200 bg-slate-50 opacity-60"
                      }`}
                    >
                      <div
                        className={`h-7 w-7 rounded-xl flex items-center justify-center shrink-0 transition-colors ${
                          done
                            ? "bg-emerald-500 text-white shadow-xs"
                            : active
                            ? "bg-[#ff5a1f] text-white animate-pulse"
                            : "bg-slate-200 text-slate-400"
                        }`}
                      >
                        {done ? (
                          <Check size={14} strokeWidth={3} />
                        ) : (
                          <span className="h-2 w-2 rounded-full bg-white" />
                        )}
                      </div>

                      <div className="min-w-0 flex-1">
                        <div className={`text-sm font-bold ${done || active ? "text-slate-800" : "text-slate-500"}`}>
                          {stage.label}
                        </div>
                        <div className="text-xs text-slate-400 truncate">{stage.desc}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {error && (
              <button
                onClick={() => router.push("/")}
                className="mt-6 inline-flex items-center gap-2 rounded-2xl bg-slate-900 hover:bg-[#ff5a1f] text-white px-6 py-3 text-sm font-semibold transition-colors cursor-pointer"
              >
                <AlertCircle size={16} />
                Back to Upload
              </button>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
