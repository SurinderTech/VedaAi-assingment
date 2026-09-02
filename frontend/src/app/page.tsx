"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Upload,
  FileText,
  X,
  ArrowRight,
  CheckCircle2,
  Clock,
  Cloud,
  Database,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { uploadAssessment, startProcessing, getApiUrl } from "@/lib/api";

function UploadCard({
  label,
  file,
  onPick,
  onClear,
}: {
  label: string;
  file: File | null;
  onPick: (f: File) => void;
  onClear: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onPick(f);
      }}
      onClick={() => !file && inputRef.current?.click()}
      className={`relative flex-1 min-h-[170px] sm:min-h-[190px] rounded-2xl border-2 border-dashed p-5 sm:p-7 flex flex-col items-center justify-center text-center transition-all duration-200 cursor-pointer ${
        dragOver
          ? "border-[#ff5a1f] bg-[#fff0ea]"
          : file
          ? "border-emerald-400 bg-emerald-50/50"
          : "border-[#d5d8e0] bg-white hover:border-[#ff5a1f] hover:bg-[#fff9f6] shadow-2xs"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPick(f);
        }}
      />

      {!file ? (
        <>
          <div className="h-10 w-10 sm:h-12 sm:w-12 rounded-xl bg-[#f1f3f7] text-slate-700 flex items-center justify-center mb-3 shadow-2xs">
            <Upload size={20} />
          </div>
          <div className="text-sm sm:text-base font-bold text-slate-800">
            Upload <span className="text-[#ff5a1f]">{label}</span>
          </div>
          <div className="text-xs font-medium text-slate-400 mt-1">Max 10MB</div>
        </>
      ) : (
        <div
          className="w-full flex items-center gap-3 rounded-xl border border-emerald-200 bg-white p-3.5 text-left shadow-2xs"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="h-10 w-10 rounded-lg bg-emerald-100 text-emerald-600 flex items-center justify-center shrink-0">
            <FileText size={18} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1">
              <span className="text-[11px] font-bold text-emerald-700 uppercase">{label}</span>
              <CheckCircle2 size={12} className="text-emerald-500" />
            </div>
            <div className="text-xs font-semibold text-slate-800 truncate">{file.name}</div>
            <div className="text-[10px] text-slate-400">{(file.size / 1024).toFixed(0)} KB</div>
          </div>
          <button
            onClick={onClear}
            className="h-7 w-7 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-700 flex items-center justify-center shrink-0"
          >
            <X size={14} />
          </button>
        </div>
      )}
    </div>
  );
}

export default function UploadPage() {
  const router = useRouter();
  const [questionPaper, setQuestionPaper] = useState<File | null>(null);
  const [answerSheet, setAnswerSheet] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const canStart = questionPaper && answerSheet && !submitting;

  async function handleStart() {
    if (!questionPaper || !answerSheet) return;
    setSubmitting(true);
    setError(null);
    try {
      console.log("[VedaAI] Uploading assessment to:", getApiUrl());
      const id = await uploadAssessment(questionPaper, answerSheet);
      await startProcessing(id);
      router.push(`/assessment/${id}/processing`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong. Please try again.");
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen w-full bg-[#f4f5f8] flex flex-col lg:flex-row font-sans">
      {/* Sidebar */}
      <Sidebar isMobileOpen={isMobileOpen} onMobileClose={() => setIsMobileOpen(false)} />

      {/* Main Full Viewport Canvas */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen bg-gradient-to-b from-white via-[#f7f8fa] to-[#e4e7ed]">
        <Header title="Exams" onMenuClick={() => setIsMobileOpen(true)} />

        <main className="flex-1 flex flex-col items-center justify-center px-4 sm:px-8 py-6 lg:py-10">
          <div className="w-full max-w-4xl flex flex-col items-center">
            {/* Headline */}
            <h1 className="text-2xl sm:text-4xl lg:text-5xl font-extrabold text-[#1a1a1a] tracking-tight leading-tight text-center">
              Upload <span className="text-[#ff5a1f] underline decoration-[#ff5a1f]/30">Question Paper &amp; Answer Sheets</span>
            </h1>
            <p className="text-xs sm:text-sm font-medium text-slate-500 mt-2.5 mb-6 sm:mb-8 text-center">
              Upload both files to get started
            </p>

            {/* Half-Body Girl Graphic Centerpiece with Orbit Ring & 4 Orbit Badges */}
            <div className="relative mb-8 sm:mb-10 flex items-center justify-center">
              {/* Translucent Orange Orbit Graphic */}
              <div className="relative h-44 w-44 sm:h-52 sm:w-52 lg:h-56 lg:w-56 rounded-full bg-gradient-to-tr from-[#ffe4d6] via-[#fff0ea] to-[#ffece2] border-2 border-[#ffc4ab] flex items-center justify-center shadow-md p-3">
                
                {/* Orbit Badge 1: Top Left Document Icon */}
                <div className="absolute top-2 left-2 h-7 w-7 sm:h-8 sm:w-8 rounded-full bg-[#ff5a1f] text-white flex items-center justify-center shadow-md border-2 border-white">
                  <FileText size={14} />
                </div>

                {/* Orbit Badge 2: Top Right Clock Icon */}
                <div className="absolute top-3 right-3 h-7 w-7 sm:h-8 sm:w-8 rounded-full bg-[#ff5a1f] text-white flex items-center justify-center shadow-md border-2 border-white">
                  <Clock size={14} />
                </div>

                {/* Orbit Badge 3: Bottom Right Cloud Icon */}
                <div className="absolute bottom-4 right-2 h-7 w-7 sm:h-8 sm:w-8 rounded-full bg-[#ff5a1f] text-white flex items-center justify-center shadow-md border-2 border-white">
                  <Cloud size={14} />
                </div>

                {/* Orbit Badge 4: Bottom Left Database Icon */}
                <div className="absolute bottom-3 left-4 h-7 w-7 sm:h-8 sm:w-8 rounded-full bg-[#ff5a1f] text-white flex items-center justify-center shadow-md border-2 border-white">
                  <Database size={14} />
                </div>

                {/* 3D Half-Body Girl Avatar holding book */}
                <div className="h-34 w-34 sm:h-40 sm:w-40 lg:h-44 lg:w-44 rounded-full overflow-hidden border-4 border-white shadow-xl bg-white flex items-center justify-center">
                  <img
                    src="/girl_3d_avatar.jpg"
                    alt="Teacher holding open book avatar"
                    className="h-full w-full object-cover object-top scale-105"
                  />
                </div>
              </div>
            </div>

            {/* Dual Upload Dropzone Cards */}
            <div className="w-full max-w-2xl grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 mb-6 sm:mb-8">
              <UploadCard
                label="Question Paper"
                file={questionPaper}
                onPick={setQuestionPaper}
                onClear={() => setQuestionPaper(null)}
              />
              <UploadCard
                label="Answer Sheet"
                file={answerSheet}
                onPick={setAnswerSheet}
                onClear={() => setAnswerSheet(null)}
              />
            </div>

            {error && (
              <div className="text-xs sm:text-sm text-red-700 bg-red-50 border border-red-200 rounded-xl px-4 py-3 mb-5 max-w-md w-full text-center">
                {error}
              </div>
            )}

            {/* Start Mapping Action Button */}
            <div className="flex flex-col items-center w-full max-w-xs">
              <button
                disabled={!canStart}
                onClick={handleStart}
                className={`w-full flex items-center justify-center gap-2 px-8 py-3.5 sm:py-4 rounded-full text-xs sm:text-sm font-bold transition-all duration-300 ${
                  canStart
                    ? "bg-[#1a1a1a] hover:bg-[#ff5a1f] text-white shadow-lg cursor-pointer hover:scale-102"
                    : "bg-[#b4b4b8] text-white cursor-not-allowed opacity-90 shadow-2xs"
                }`}
              >
                <span>{submitting ? "Starting..." : "Start Mapping"}</span>
                <ArrowRight size={16} />
              </button>
              <p className="text-[11px] sm:text-xs text-slate-400 font-medium mt-3 text-center">
                Once both files are uploaded, you&rsquo;ll be able to map answers with questions
              </p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
