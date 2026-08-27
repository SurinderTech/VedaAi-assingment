"use client";

import { useState } from "react";
import { Bell, X, Check, FileCheck, AlertTriangle, Sparkles } from "lucide-react";

export interface AppNotification {
  id: string;
  title: string;
  message: string;
  time: string;
  type: "success" | "warning" | "info";
  read: boolean;
}

interface NotificationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  notifications: AppNotification[];
  onMarkAllRead: () => void;
}

export default function NotificationDrawer({
  isOpen,
  onClose,
  notifications,
  onMarkAllRead,
}: NotificationDrawerProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-900/30 backdrop-blur-xs">
      <div className="w-full max-w-sm bg-white h-full shadow-2xl flex flex-col border-l border-slate-200 animate-in slide-in-from-right duration-250">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bell size={18} className="text-[#ff5a1f]" />
            <h2 className="text-sm font-bold text-slate-900">Notifications</h2>
            <span className="px-2 py-0.5 rounded-full bg-[#fff0ea] text-[#ff5a1f] text-[11px] font-bold">
              {notifications.filter((n) => !n.read).length} new
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Actions bar */}
        {notifications.length > 0 && (
          <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-xs font-semibold text-slate-600">
            <span>Recent Assessment Events</span>
            <button
              onClick={onMarkAllRead}
              className="text-[#ff5a1f] hover:underline flex items-center gap-1 cursor-pointer"
            >
              <Check size={13} /> Mark all read
            </button>
          </div>
        )}

        {/* Notification Items */}
        <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-2">
          {notifications.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400">
              <Bell size={32} className="mb-2 text-slate-300" />
              <p className="text-xs font-medium">No new notifications. Upload an assessment to get started!</p>
            </div>
          ) : (
            notifications.map((n) => (
              <div
                key={n.id}
                className={`p-3 rounded-2xl border transition-all ${
                  n.read
                    ? "bg-white border-slate-100 text-slate-600"
                    : "bg-[#fff0ea]/50 border-[#ff5a1f]/30 text-slate-900 shadow-2xs"
                }`}
              >
                <div className="flex items-start gap-2.5">
                  <div
                    className={`h-7 w-7 rounded-xl flex items-center justify-center shrink-0 mt-0.5 ${
                      n.type === "success"
                        ? "bg-emerald-100 text-emerald-600"
                        : n.type === "warning"
                        ? "bg-amber-100 text-amber-600"
                        : "bg-[#fff0ea] text-[#ff5a1f]"
                    }`}
                  >
                    {n.type === "success" ? (
                      <FileCheck size={14} />
                    ) : n.type === "warning" ? (
                      <AlertTriangle size={14} />
                    ) : (
                      <Sparkles size={14} />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between mb-0.5">
                      <h4 className="text-xs font-bold text-slate-900 truncate">{n.title}</h4>
                      <span className="text-[10px] text-slate-400 shrink-0">{n.time}</span>
                    </div>
                    <p className="text-[11px] text-slate-600 leading-normal">{n.message}</p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
