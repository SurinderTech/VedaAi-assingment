"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import { Region } from "@/types/assessment";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw, Eye } from "lucide-react";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface Props {
  fileUrl: string;
  isPdf: boolean;
  questionNumber?: string;
  regions: Region[]; // regions to highlight for the selected question
  pageSizes: number[][]; // original [width, height] per page, matching bbox coordinate space
  activePage: number;
  onPageChange: (page: number) => void;
  totalPages: number;
}

export default function AnswerSheetViewer({
  fileUrl,
  isPdf,
  questionNumber = "",
  regions,
  pageSizes,
  activePage,
  onPageChange,
  totalPages,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderedSize, setRenderedSize] = useState<{ w: number; h: number } | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState<number>(1);

  // Jump to the first region's page whenever the selected question changes.
  useEffect(() => {
    if (regions.length > 0) {
      onPageChange(regions[0].page);
    }
  }, [regions, onPageChange]);

  const original = pageSizes && pageSizes[activePage - 1] ? pageSizes[activePage - 1] : null;
  const origW = (original && original[0]) || naturalSize?.w || 1;
  const origH = (original && original[1]) || naturalSize?.h || 1;
  const scaleX = renderedSize ? renderedSize.w / origW : 1;
  const scaleY = renderedSize ? renderedSize.h / origH : 1;
  const regionsOnPage = regions.filter((r) => r.page === activePage);

  // Compute unified outer merged bounding box covering the entire answer text
  const mergedBounds = useMemo(() => {
    if (!regionsOnPage || regionsOnPage.length === 0) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const r of regionsOnPage) {
      if (r.bbox.x < minX) minX = r.bbox.x;
      if (r.bbox.y < minY) minY = r.bbox.y;
      if (r.bbox.x + r.bbox.width > maxX) maxX = r.bbox.x + r.bbox.width;
      if (r.bbox.y + r.bbox.height > maxY) maxY = r.bbox.y + r.bbox.height;
    }
    const padX = 10;
    const padY = 8;
    return {
      x: Math.max(0, minX - padX),
      y: Math.max(0, minY - padY),
      width: (maxX - minX) + (padX * 2),
      height: (maxY - minY) + (padY * 2),
    };
  }, [regionsOnPage]);

  const baseWidth = 640;
  const currentWidth = baseWidth * zoom;

  const allSpannedPages = useMemo(() => {
    return Array.from(new Set(regions.map((r) => r.page))).sort((a, b) => a - b);
  }, [regions]);

  const nextPageInSpanned = useMemo(() => {
    return allSpannedPages.find((p) => p > activePage);
  }, [allSpannedPages, activePage]);

  const prevPageInSpanned = useMemo(() => {
    const prevs = allSpannedPages.filter((p) => p < activePage);
    return prevs.length > 0 ? prevs[prevs.length - 1] : undefined;
  }, [allSpannedPages, activePage]);

  // Scroll bounding box into view when activePage or mergedBounds changes
  useEffect(() => {
    if (mergedBounds && containerRef.current) {
      const targetY = mergedBounds.y * scaleY;
      const scrollParent = containerRef.current.parentElement;
      if (scrollParent) {
        scrollParent.scrollTo({
          top: Math.max(0, targetY - 60),
          behavior: "smooth",
        });
      }
    }
  }, [mergedBounds, scaleY, activePage]);

  return (
    <div className="flex flex-col h-full bg-[#eef0f4]">
      {/* Control Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-white border-b border-slate-200 shadow-xs z-10 text-xs font-semibold text-slate-700">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="h-6 w-6 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
            <Eye size={14} />
          </div>
          <span>Student Answer Sheet</span>
          {regionsOnPage.length > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 text-[11px] font-bold">
              {regionsOnPage.length} region{regionsOnPage.length > 1 ? "s" : ""} mapped
            </span>
          )}
          {allSpannedPages.length > 1 && (
            <div className="flex items-center gap-1 bg-amber-50 border border-amber-200 rounded-full px-2.5 py-0.5 text-[11px] font-bold text-amber-800">
              <span>Spans Pages:</span>
              {allSpannedPages.map((p) => (
                <button
                  key={p}
                  onClick={() => onPageChange(p)}
                  className={`px-2 py-0.5 rounded-full cursor-pointer transition-all ${
                    p === activePage
                      ? "bg-amber-600 text-white shadow-xs scale-105"
                      : "bg-amber-100 text-amber-900 hover:bg-amber-200"
                  }`}
                  title={`Jump to Page ${p}`}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Zoom controls */}
          <div className="flex items-center bg-slate-100 rounded-lg p-0.5 border border-slate-200">
            <button
              onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
              className="p-1 hover:bg-white rounded text-slate-600 transition-colors"
              title="Zoom out"
            >
              <ZoomOut size={14} />
            </button>
            <span className="px-2 text-[11px] font-mono font-bold text-slate-600">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(2, z + 0.25))}
              className="p-1 hover:bg-white rounded text-slate-600 transition-colors"
              title="Zoom in"
            >
              <ZoomIn size={14} />
            </button>
            {zoom !== 1 && (
              <button
                onClick={() => setZoom(1)}
                className="p-1 hover:bg-white rounded text-slate-400 hover:text-slate-700 transition-colors"
                title="Reset zoom"
              >
                <RotateCcw size={13} />
              </button>
            )}
          </div>

          {/* Page navigation */}
          <div className="flex items-center gap-1.5 bg-slate-100 rounded-lg p-0.5 border border-slate-200">
            <button
              disabled={activePage <= 1}
              onClick={() => onPageChange(activePage - 1)}
              className="p-1 rounded hover:bg-white disabled:opacity-30 disabled:hover:bg-transparent text-slate-600 transition-colors cursor-pointer"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="px-1 text-[11px] font-semibold text-slate-600">
              Page {activePage} of {totalPages}
            </span>
            <button
              disabled={activePage >= totalPages}
              onClick={() => onPageChange(activePage + 1)}
              className="p-1 rounded hover:bg-white disabled:opacity-30 disabled:hover:bg-transparent text-slate-600 transition-colors cursor-pointer"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Canvas / Image Display */}
      <div className="flex-1 overflow-auto scrollbar-thin flex justify-center py-8 px-16 relative">
        <div ref={containerRef} className="relative inline-block shadow-2xl rounded-xl bg-white border border-slate-300">
          {isPdf ? (
            <Document
              file={fileUrl}
              loading={
                <div className="p-16 text-center text-xs font-medium text-slate-400 flex flex-col items-center gap-2">
                  <div className="h-6 w-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                  Loading document page...
                </div>
              }
            >
              <Page
                pageNumber={activePage}
                width={currentWidth}
                onRenderSuccess={(page) => {
                  setNaturalSize({ w: page.originalWidth, h: page.originalHeight });
                  setRenderedSize({ w: page.width, h: page.height });
                }}
              />
            </Document>
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={fileUrl}
              alt="Handwritten answer sheet"
              style={{ width: currentWidth }}
              onLoad={(e) => {
                const img = e.currentTarget;
                setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
                setRenderedSize({ w: img.clientWidth, h: img.clientHeight });
              }}
            />
          )}

          {/* Single Unified Answer Bounding Box & Left-Margin Question Badge (Image 2 style) */}
          {renderedSize && mergedBounds && (
            <div
              className="absolute border-2 border-emerald-500 bg-emerald-500/10 rounded-2xl transition-all duration-300 pointer-events-none z-20 shadow-[0_0_20px_rgba(16,185,129,0.18)] ring-1 ring-emerald-400/30"
              style={{
                left: mergedBounds.x * scaleX,
                top: mergedBounds.y * scaleY,
                width: mergedBounds.width * scaleX,
                height: mergedBounds.height * scaleY,
              }}
            >
              {/* Question Badge sitting cleanly in the left margin / empty space outside answer text */}
              <div
                className="absolute top-0 -left-2.5 -translate-x-full z-30 flex items-center justify-center bg-emerald-600 text-white text-xs md:text-sm font-bold px-2.5 py-0.5 rounded-lg shadow-md border border-emerald-400/40 select-none whitespace-nowrap"
              >
                <span>
                  {(() => {
                    const qStr = (questionNumber || "").trim();
                    if (!qStr) return "Q";
                    if (qStr.toLowerCase() === "unmatched") return "Unmatched";
                    if (/^\d/.test(qStr)) return `Q${qStr}`;
                    if (/^q[\.\s]?/i.test(qStr)) return qStr.replace(/^q[\.\s]*/i, "Q");
                    return qStr.startsWith("Q") ? qStr : `Q${qStr}`;
                  })()}
                </span>
              </div>
            </div>
          )}

          {/* Floating Next Page Jump Banner when answer continues on next page */}
          {renderedSize && mergedBounds && nextPageInSpanned && (
            <div
              className="absolute z-40 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold px-3.5 py-2 rounded-xl shadow-2xl flex items-center gap-2.5 cursor-pointer transition-all border border-emerald-400/50 animate-bounce -translate-x-1/2 left-1/2"
              style={{
                top: Math.min(renderedSize.h - 48, (mergedBounds.y + mergedBounds.height) * scaleY + 12),
              }}
              onClick={() => onPageChange(nextPageInSpanned)}
              title={`Jump directly to Page ${nextPageInSpanned}`}
            >
              <span>Q{questionNumber || "Answer"} continues on Page {nextPageInSpanned}</span>
              <span className="bg-white/25 px-2.5 py-1 rounded-lg text-[11px] font-mono flex items-center gap-1">
                Jump to Page {nextPageInSpanned} <ChevronRight size={13} />
              </span>
            </div>
          )}

          {/* Floating Previous Page Back Banner when viewing continuation page */}
          {renderedSize && mergedBounds && prevPageInSpanned && (
            <div
              className="absolute z-40 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold px-3.5 py-2 rounded-xl shadow-2xl flex items-center gap-2.5 cursor-pointer transition-all border border-emerald-400/50 -translate-x-1/2 left-1/2"
              style={{
                top: Math.max(8, (mergedBounds.y * scaleY) - 44),
              }}
              onClick={() => onPageChange(prevPageInSpanned)}
              title={`Return to Page ${prevPageInSpanned}`}
            >
              <span className="bg-white/25 px-2.5 py-1 rounded-lg text-[11px] font-mono flex items-center gap-1">
                <ChevronLeft size={13} /> Back to Page {prevPageInSpanned}
              </span>
              <span>Q{questionNumber || "Answer"} continued from Page {prevPageInSpanned}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

