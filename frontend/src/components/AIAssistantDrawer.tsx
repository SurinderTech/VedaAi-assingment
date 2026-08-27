"use client";

import { useState } from "react";
import { Sparkles, X, Send, AlertTriangle, HelpCircle, CheckCircle2, ArrowRight } from "lucide-react";
import { askAssistant, AssistantResponse } from "@/lib/api";

interface AIAssistantDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  assessmentId?: string | null;
  selectedQuestionId?: string | null;
  onSelectQuestion?: (questionId: string) => void;
}

interface Message {
  sender: "user" | "assistant";
  text: string;
  responseObj?: AssistantResponse;
}

export default function AIAssistantDrawer({
  isOpen,
  onClose,
  assessmentId = null,
  selectedQuestionId = null,
  onSelectQuestion,
}: AIAssistantDrawerProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "assistant",
      text: "Hi! I'm VedaAI Teacher Assistant. How can I help you evaluate this assessment?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  async function handleSend(textToSend?: string) {
    const q = textToSend || input;
    if (!q.trim() || loading) return;

    setMessages((prev) => [...prev, { sender: "user", text: q }]);
    if (!textToSend) setInput("");
    setLoading(true);

    try {
      const res = await askAssistant(assessmentId, q, selectedQuestionId || undefined);
      setMessages((prev) => [...prev, { sender: "assistant", text: res.reply, responseObj: res }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          sender: "assistant",
          text: "I analyzed the current assessment context. Please review the highlighted regions on the sheet.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-900/30 backdrop-blur-xs">
      <div className="w-full max-w-md bg-white h-full shadow-2xl flex flex-col border-l border-slate-200 animate-in slide-in-from-right duration-250">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 bg-slate-900 text-white flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-xl bg-[#ff5a1f] flex items-center justify-center text-white shadow-sm">
              <Sparkles size={18} />
            </div>
            <div>
              <h2 className="text-sm font-bold tracking-tight">VedaAI Assistant</h2>
              <p className="text-[10px] text-slate-300">Context-Aware Exam Co-Pilot</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/10 text-slate-300 hover:text-white transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Action Presets */}
        <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center gap-1.5 overflow-x-auto scrollbar-thin">
          <button
            onClick={() => handleSend("What needs my attention?")}
            className="px-2.5 py-1 rounded-full bg-white border border-slate-200 hover:border-[#ff5a1f] hover:text-[#ff5a1f] text-[11px] font-bold text-slate-700 shrink-0 transition-colors cursor-pointer flex items-center gap-1"
          >
            <AlertTriangle size={12} className="text-amber-500" /> What needs attention?
          </button>
          <button
            onClick={() => handleSend("Show unanswered questions")}
            className="px-2.5 py-1 rounded-full bg-white border border-slate-200 hover:border-[#ff5a1f] hover:text-[#ff5a1f] text-[11px] font-bold text-slate-700 shrink-0 transition-colors cursor-pointer flex items-center gap-1"
          >
            <HelpCircle size={12} className="text-slate-400" /> Unanswered questions
          </button>
          <button
            onClick={() => handleSend("Review low-confidence mappings")}
            className="px-2.5 py-1 rounded-full bg-white border border-slate-200 hover:border-[#ff5a1f] hover:text-[#ff5a1f] text-[11px] font-bold text-slate-700 shrink-0 transition-colors cursor-pointer flex items-center gap-1"
          >
            <CheckCircle2 size={12} className="text-emerald-500" /> Review mappings
          </button>
        </div>

        {/* Message Stream */}
        <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-3.5">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex flex-col ${m.sender === "user" ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed ${
                  m.sender === "user"
                    ? "bg-[#ff5a1f] text-white rounded-br-2xs shadow-xs"
                    : "bg-slate-100 text-slate-800 rounded-bl-2xs border border-slate-200/60"
                }`}
              >
                <div className="whitespace-pre-wrap">{m.text}</div>

                {/* Clickable Question Badges */}
                {m.responseObj && onSelectQuestion && (
                  <div className="mt-2.5 pt-2 border-t border-slate-200/60 space-y-1.5">
                    {m.responseObj.unanswered_questions.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="text-[10px] font-bold text-slate-500">Jump to Unanswered:</span>
                        {m.responseObj.unanswered_questions.map((q) => (
                          <button
                            key={q}
                            onClick={() => {
                              onSelectQuestion(q);
                              onClose();
                            }}
                            className="px-2 py-0.5 rounded-md bg-slate-200 hover:bg-[#ff5a1f] hover:text-white text-[11px] font-bold text-slate-700 transition-colors cursor-pointer flex items-center gap-0.5"
                          >
                            Q{q} <ArrowRight size={10} />
                          </button>
                        ))}
                      </div>
                    )}
                    {m.responseObj.review_questions.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="text-[10px] font-bold text-slate-500">Jump to Review:</span>
                        {m.responseObj.review_questions.map((q) => (
                          <button
                            key={q}
                            onClick={() => {
                              onSelectQuestion(q);
                              onClose();
                            }}
                            className="px-2 py-0.5 rounded-md bg-amber-100 hover:bg-[#ff5a1f] hover:text-white text-[11px] font-bold text-amber-800 transition-colors cursor-pointer flex items-center gap-0.5"
                          >
                            Q{q} <ArrowRight size={10} />
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-xs text-slate-400 font-medium">
              <div className="h-2 w-2 rounded-full bg-[#ff5a1f] animate-ping" />
              VedaAI Assistant is analyzing assessment...
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-3 border-t border-slate-200 bg-white">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              placeholder="Ask VedaAI about this assessment..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-1 px-3.5 py-2 rounded-xl bg-slate-100 border border-slate-200 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-[#ff5a1f]"
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="h-9 w-9 rounded-xl bg-[#ff5a1f] hover:bg-[#e04810] text-white flex items-center justify-center disabled:opacity-40 transition-colors cursor-pointer"
            >
              <Send size={15} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
