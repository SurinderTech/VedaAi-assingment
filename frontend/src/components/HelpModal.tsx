"use client";

import { X, HelpCircle, FileText, CheckCircle2, Sparkles, Eye, ShieldCheck } from "lucide-react";

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function HelpModal({ isOpen, onClose }: HelpModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
      <div className="bg-white rounded-3xl p-6 sm:p-8 max-w-lg w-full border border-slate-200 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto scrollbar-thin">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-2xl bg-[#fff0ea] text-[#ff5a1f] flex items-center justify-center font-bold">
              <HelpCircle size={20} />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">How VedaAI Works</h2>
              <p className="text-xs text-slate-400">Examiner Platform Guide &amp; FAQ</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content sections */}
        <div className="space-y-4 text-xs text-slate-600 leading-relaxed">
          <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/60 space-y-1">
            <div className="flex items-center gap-2 font-bold text-slate-900">
              <FileText size={15} className="text-[#ff5a1f]" />
              <span>1. Upload Question Paper &amp; Answer Sheet</span>
            </div>
            <p className="text-slate-500">
              Upload a typed question paper (PDF or image) and a student&rsquo;s handwritten answer sheet (PDF or image).
            </p>
          </div>

          <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/60 space-y-1">
            <div className="flex items-center gap-2 font-bold text-slate-900">
              <Sparkles size={15} className="text-[#ff5a1f]" />
              <span>2. PaddleOCR &amp; SentenceTransformer Mapping</span>
            </div>
            <p className="text-slate-500">
              PaddleOCR reads handwritten text and bounding boxes. SentenceTransformer embeddings (all-MiniLM-L6-v2) match student answers to questions 1-to-1.
            </p>
          </div>

          <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/60 space-y-1">
            <div className="flex items-center gap-2 font-bold text-slate-900">
              <Eye size={15} className="text-[#ff5a1f]" />
              <span>3. Exact Highlight Regions</span>
            </div>
            <p className="text-slate-500">
              Click any question in the left panel to jump directly to the highlighted answer region on the student&rsquo;s sheet.
            </p>
          </div>

          <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/60 space-y-1">
            <div className="flex items-center gap-2 font-bold text-slate-900">
              <ShieldCheck size={15} className="text-emerald-600" />
              <span>4. Subquestions &amp; Edge Cases</span>
            </div>
            <p className="text-slate-500">
              Subquestions like 11(a) and 11(b) are parsed independently. Unanswered questions and out-of-order answers are correctly identified.
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="w-full py-3 rounded-2xl bg-slate-900 hover:bg-[#ff5a1f] text-white text-xs font-bold transition-colors cursor-pointer"
        >
          Got it!
        </button>
      </div>
    </div>
  );
}
