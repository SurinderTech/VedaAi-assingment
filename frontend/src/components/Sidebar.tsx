"use client";

import { useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  LayoutGrid,
  Users,
  FileText,
  FileCheck,
  Bookmark,
  Settings,
  Sparkles,
  PanelLeft,
  X,
} from "lucide-react";
import VedaLogo from "./VedaLogo";

interface SidebarProps {
  isMobileOpen?: boolean;
  onMobileClose?: () => void;
}

export default function Sidebar({ isMobileOpen = false, onMobileClose }: SidebarProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [showToolkitModal, setShowToolkitModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  const NAV = [
    { icon: LayoutGrid, label: "Home", path: "/" },
    { icon: Users, label: "My Classroom", path: "/classroom" },
    { icon: FileText, label: "Assignments", path: "/assignments" },
    { icon: FileCheck, label: "Exams", path: "/" },
    { icon: Bookmark, label: "My Library", path: "/library" },
  ];

  const handleNavClick = (path: string) => {
    if (onMobileClose) onMobileClose();
    router.push(path);
  };

  const content = (
    <div className="flex flex-col justify-between h-full bg-white text-slate-800 p-5 border-r border-slate-200/80">
      <div>
        {/* Header Logo + Sidebar Collapse Button */}
        <div className="flex items-center justify-between mb-7">
          <div className="cursor-pointer" onClick={() => handleNavClick("/")}>
            <VedaLogo size="md" />
          </div>
          {onMobileClose ? (
            <button
              onClick={onMobileClose}
              className="lg:hidden text-slate-400 hover:text-slate-700 p-1 rounded-lg hover:bg-slate-100 transition-colors cursor-pointer"
            >
              <X size={20} />
            </button>
          ) : (
            <button className="hidden lg:block text-slate-400 hover:text-slate-700 p-1 transition-colors cursor-pointer">
              <PanelLeft size={18} />
            </button>
          )}
        </div>

        {/* AI Teacher's Toolkit Button */}
        <button
          onClick={() => setShowToolkitModal(true)}
          className="w-full flex items-center justify-center gap-2.5 rounded-full bg-[#252528] hover:bg-[#1a1a1c] border-[2.5px] border-[#ff5a1f] px-5 py-3 text-xs sm:text-sm font-bold text-white mb-8 shadow-sm hover:opacity-95 transition-all cursor-pointer group"
        >
          <div className="flex items-center gap-0.5 text-white">
            <Sparkles size={16} className="fill-white text-white" />
            <Sparkles size={12} className="fill-white text-white -ml-1 -mt-1" />
          </div>
          <span className="text-white tracking-tight">AI Teacher&rsquo;s Toolkit</span>
        </button>

        {/* Navigation Items */}
        <nav className="space-y-1.5">
          {NAV.map(({ icon: Icon, label, path }) => {
            const isActive = pathname === path || (path === "/" && (pathname === "/exams" || pathname === "/"));
            return (
              <button
                key={label}
                onClick={() => handleNavClick(path)}
                className={`w-full flex items-center gap-3.5 px-4 py-3 rounded-xl text-xs sm:text-sm font-semibold cursor-pointer transition-all ${
                  isActive
                    ? "bg-[#efefef] text-[#1a1a1a] shadow-2xs font-extrabold"
                    : "text-slate-400 hover:bg-slate-50 hover:text-slate-700"
                }`}
              >
                <Icon size={18} className={isActive ? "text-[#1a1a1a]" : "text-slate-400"} />
                <span>{label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Settings & School Crest Card */}
      <div className="space-y-4 pt-4 border-t border-slate-100">
        <button
          onClick={() => setShowSettingsModal(true)}
          className="w-full flex items-center gap-3 px-3 py-1.5 text-slate-400 text-xs sm:text-sm font-medium cursor-pointer hover:text-slate-700 transition-colors"
        >
          <Settings size={18} />
          <span>Settings</span>
        </button>

        {/* Delhi Public School Crest Badge */}
        <div className="flex items-center gap-3 rounded-2xl bg-[#f4f4f6] p-3 border border-slate-200/50">
          <div className="h-9 w-9 rounded-full bg-emerald-700 text-white flex items-center justify-center font-bold text-xs shrink-0 border-2 border-emerald-800 shadow-2xs">
            <svg className="w-5 h-5 fill-current text-white" viewBox="0 0 24 24">
              <path d="M12 2L4 5v6c0 5.55 3.84 10.74 8 12 4.16-1.26 8-6.45 8-12V5l-8-3zm0 4a3 3 0 110 6 3 3 0 010-6z" />
            </svg>
          </div>
          <div className="text-xs leading-tight min-w-0">
            <div className="font-bold text-slate-900 truncate">Delhi Public School</div>
            <div className="text-slate-400 text-[11px] font-medium truncate">Bokaro Steel City</div>
          </div>
        </div>
      </div>

      {/* Toolkit Modal */}
      {showToolkitModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
          <div className="bg-white rounded-3xl p-6 max-w-sm w-full border border-slate-200 shadow-2xl text-center space-y-4">
            <div className="mx-auto h-12 w-12 rounded-2xl bg-[#fff0ea] text-[#ff5a1f] flex items-center justify-center">
              <Sparkles size={24} />
            </div>
            <h3 className="text-base font-bold text-slate-900">AI Teacher&rsquo;s Toolkit</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Automated OCR question extraction, handwritten response mapping, confidence scoring, and AI evaluation assistant active.
            </p>
            <button
              onClick={() => setShowToolkitModal(false)}
              className="w-full py-2.5 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-[#ff5a1f] transition-colors cursor-pointer"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettingsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
          <div className="bg-white rounded-3xl p-6 max-w-sm w-full border border-slate-200 shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-slate-900">Workspace Settings</h3>
            <div className="space-y-2 text-xs text-slate-600">
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="font-medium">Primary LLM Provider</span>
                <span className="font-bold text-[#ff5a1f]">Gemini 2.0 Flash</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="font-medium">Fallback Providers</span>
                <span>Groq, OpenRouter</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="font-medium">OCR Engine</span>
                <span className="font-bold text-emerald-600">PaddleOCR</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="font-medium">Embeddings</span>
                <span className="font-bold text-purple-600">all-MiniLM-L6-v2</span>
              </div>
              <div className="flex justify-between py-1.5">
                <span className="font-medium">Exact BBox Highlighting</span>
                <span className="font-bold text-emerald-600">Active</span>
              </div>
            </div>
            <button
              onClick={() => setShowSettingsModal(false)}
              className="w-full py-2.5 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-[#ff5a1f] transition-colors cursor-pointer"
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Desktop Persistent Sidebar */}
      <aside className="hidden lg:block w-64 shrink-0 h-screen sticky top-0 z-20">
        {content}
      </aside>

      {/* Mobile / Tablet Overlay Drawer */}
      {isMobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity"
            onClick={onMobileClose}
          />
          <div className="relative w-72 max-w-[80vw] h-full shadow-2xl z-10 animate-in slide-in-from-left duration-200">
            {content}
          </div>
        </div>
      )}
    </>
  );
}
