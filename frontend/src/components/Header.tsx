"use client";

import React from "react";
import { Menu, ArrowLeft, FileCheck, HelpCircle, Bell, Sparkles, ChevronDown } from "lucide-react";
import VedaLogo from "./VedaLogo";

interface HeaderProps {
  title?: string;
  onMenuClick?: () => void;
}

export default function Header({ title = "Exams", onMenuClick }: HeaderProps) {
  return (
    <header className="h-16 px-4 sm:px-8 flex items-center justify-between sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-2xs">
      {/* Left Menu / Back Button + Title */}
      <div className="flex items-center gap-2 sm:gap-3">
        {onMenuClick && (
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 rounded-xl text-slate-700 hover:bg-slate-100 transition-colors"
            title="Open Menu"
          >
            <Menu size={20} />
          </button>
        )}

        <div className="lg:hidden">
          <VedaLogo size="sm" showText={false} />
        </div>

        <button className="hidden sm:flex h-8 w-8 rounded-full hover:bg-slate-100 items-center justify-center text-slate-700 transition-colors cursor-pointer">
          <ArrowLeft size={18} />
        </button>

        <div className="flex items-center gap-2 text-slate-600 font-semibold text-xs sm:text-sm">
          <FileCheck size={16} className="text-slate-400 hidden sm:inline" />
          <span className="text-slate-700 font-bold">{title}</span>
        </div>
      </div>

      {/* Right User Controls & Profile */}
      <div className="flex items-center gap-2 sm:gap-4 text-slate-600">
        {/* Help icon */}
        <button className="p-1.5 rounded-full hover:bg-slate-100 text-slate-600 transition-colors hidden sm:block">
          <HelpCircle size={18} />
        </button>

        {/* Notification bell with red indicator */}
        <button className="p-1.5 rounded-full hover:bg-slate-100 text-slate-600 transition-colors relative">
          <Bell size={18} />
          <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-[#ff5a1f]" />
        </button>

        {/* AI Sparkle Icon */}
        <button className="p-1.5 rounded-full hover:bg-slate-100 text-slate-600 transition-colors hidden sm:block">
          <Sparkles size={18} className="text-slate-700" />
        </button>

        {/* User Profile avatar + Name */}
        <div className="flex items-center gap-2 pl-1 sm:pl-2 cursor-pointer group">
          <div className="h-8 w-8 rounded-full bg-slate-900 border border-slate-200 overflow-hidden shadow-2xs">
            <img
              src="/girl_3d_avatar.jpg"
              alt="Madhur Rastogi"
              className="h-full w-full object-cover"
            />
          </div>
          <span className="text-xs font-bold text-slate-800 hidden md:inline">
            Madhur Rastogi
          </span>
          <ChevronDown size={14} className="text-slate-400 group-hover:text-slate-700 transition-colors hidden sm:block" />
        </div>
      </div>
    </header>
  );
}
