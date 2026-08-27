"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Sparkles, AlertCircle } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { getStatus } from "@/lib/api";
import { AssessmentStatus } from "@/types/assessment";

export default function ProcessingPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [status, setStatus] = useState<AssessmentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    let failCount = 0;

    async function poll() {
      try {
        const s = await getStatus(params.id);
        if (cancelled) return;
        setStatus(s);
        failCount = 0;
        if (s.state === "completed") {
          router.push(`/assessment/${params.id}/workspace`);
          return;
        }
        if (s.state === "failed") {
          setError(s.message || "Processing failed.");
          return;
        }
        timer = setTimeout(poll, 1200);
      } catch (err: any) {
        if (!cancelled) {
          failCount++;
          if (err?.message?.includes("404") || failCount >= 4) {
            setError("Assessment session not found or server restarted. Please re-upload your document.");
          } else {
            timer = setTimeout(poll, 1500);
          }
        }
      }
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [params.id, router]);

  return (
    <div className="min-h-screen w-full bg-[#f4f5f8] flex flex-col lg:flex-row font-sans">
      <Sidebar isMobileOpen={isMobileOpen} onMobileClose={() => setIsMobileOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0 min-h-screen bg-gradient-to-b from-white via-[#f7f8fa] to-[#e4e7ed]">
        <Header title="Assessment Processing" onMenuClick={() => setIsMobileOpen(true)} />

        <main className="flex-1 flex items-center justify-center px-4 py-12">
          <div className="w-full max-w-xl bg-white rounded-[2.5rem] border border-slate-200/90 p-10 sm:p-14 shadow-2xl shadow-slate-900/5 text-center flex flex-col items-center justify-center min-h-[400px]">
            {!error ? (
              <>
                {/* Floating glowing orange/red sparkles icon matching Figma Extraction flow */}
                <div className="relative mb-8 flex items-center justify-center">
                  <div className="absolute -inset-4 rounded-full bg-[#ff5a1f]/15 animate-ping" style={{ animationDuration: "2.5s" }} />
                  <div className="relative h-20 w-20 rounded-3xl bg-gradient-to-br from-[#ff7a45] to-[#ff5a1f] text-white flex items-center justify-center shadow-xl shadow-[#ff5a1f]/30">
                    <div className="relative flex items-center justify-center">
                      <Sparkles size={42} className="text-white fill-white animate-pulse" />
                    </div>
                  </div>
                </div>

                <h1 className="text-2xl font-black text-slate-900 mb-2 tracking-tight">Extracting...</h1>
                <p className="text-sm font-semibold text-slate-400 mb-6">This may take a while</p>

                {/* Progress bar */}
                <div className="w-full max-w-xs bg-slate-100 h-2 rounded-full overflow-hidden mb-3">
                  <div
                    className="bg-gradient-to-r from-[#ff7a45] to-[#ff5a1f] h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.max(15, (status?.progress || 0.1) * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 font-mono font-medium">
                  {status?.message || "Running OCR & 1-to-1 Answer Mapping..."}
                </p>
              </>
            ) : (
              <>
                <div className="h-16 w-16 rounded-2xl bg-rose-100 text-rose-600 flex items-center justify-center mb-4">
                  <AlertCircle size={32} />
                </div>
                <h1 className="text-xl font-bold text-slate-900 mb-1">Processing Error</h1>
                <p className="text-xs text-slate-500 max-w-sm mb-6">{error}</p>
                <button
                  onClick={() => router.push("/")}
                  className="px-6 py-3 rounded-2xl bg-slate-900 hover:bg-[#ff5a1f] text-white text-xs font-bold transition-colors cursor-pointer"
                >
                  Back to Upload
                </button>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
