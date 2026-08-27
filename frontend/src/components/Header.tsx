"use client";

import React, { useState } from "react";
import { Menu, ArrowLeft, FileCheck, HelpCircle, Bell, Sparkles, ChevronDown } from "lucide-react";
import VedaLogo from "./VedaLogo";
import AIAssistantDrawer from "./AIAssistantDrawer";
import NotificationDrawer, { AppNotification } from "./NotificationDrawer";
import HelpModal from "./HelpModal";

interface HeaderProps {
  title?: string;
  onMenuClick?: () => void;
  assessmentId?: string | null;
  selectedQuestionId?: string | null;
  onSelectQuestion?: (questionId: string) => void;
}

export default function Header({
  title = "Exams",
  onMenuClick,
  assessmentId = null,
  selectedQuestionId = null,
  onSelectQuestion,
}: HeaderProps) {
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const [notifications, setNotifications] = useState<AppNotification[]>([
    {
      id: "1",
      title: "Assessment Ready",
      message: "PaddleOCR & SentenceTransformers pipeline initialized.",
      time: "Just now",
      type: "success",
      read: false,
    },
    {
      id: "2",
      title: "AI Co-Pilot Active",
      message: "VedaAI context-aware assistant connected to Gemini.",
      time: "2m ago",
      type: "info",
      read: false,
    },
  ]);

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <>
      <header className="h-16 px-4 sm:px-8 flex items-center justify-between sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-2xs">
        {/* Left Menu / Back Button + Title */}
        <div className="flex items-center gap-2 sm:gap-3">
          {onMenuClick && (
            <button
              onClick={onMenuClick}
              className="lg:hidden p-2 rounded-xl text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
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
        <div className="flex items-center gap-2 sm:gap-3.5 text-slate-600">
          {/* Help icon */}
          <button
            onClick={() => setIsHelpOpen(true)}
            className="p-2 rounded-xl hover:bg-slate-100 text-slate-600 transition-colors cursor-pointer"
            title="Help & FAQ"
          >
            <HelpCircle size={18} />
          </button>

          {/* Notification bell with red indicator */}
          <button
            onClick={() => setIsNotificationsOpen(true)}
            className="p-2 rounded-xl hover:bg-slate-100 text-slate-600 transition-colors relative cursor-pointer"
            title="Notifications"
          >
            <Bell size={18} />
            {unreadCount > 0 && (
              <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-[#ff5a1f]" />
            )}
          </button>

          {/* AI Sparkle Icon button */}
          <button
            onClick={() => setIsAssistantOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#fff0ea] border border-[#ff5a1f]/30 hover:border-[#ff5a1f] text-[#ff5a1f] font-bold text-xs shadow-2xs transition-all cursor-pointer"
            title="Open VedaAI Teacher Assistant"
          >
            <Sparkles size={15} className="animate-spin" style={{ animationDuration: "6s" }} />
            <span className="hidden sm:inline">VedaAI Assistant</span>
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

      {/* AI Assistant Drawer */}
      <AIAssistantDrawer
        isOpen={isAssistantOpen}
        onClose={() => setIsAssistantOpen(false)}
        assessmentId={assessmentId}
        selectedQuestionId={selectedQuestionId}
        onSelectQuestion={onSelectQuestion}
      />

      {/* Notification Drawer */}
      <NotificationDrawer
        isOpen={isNotificationsOpen}
        onClose={() => setIsNotificationsOpen(false)}
        notifications={notifications}
        onMarkAllRead={() => setNotifications((prev) => prev.map((n) => ({ ...n, read: true })))}
      />

      {/* Help Modal */}
      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </>
  );
}
