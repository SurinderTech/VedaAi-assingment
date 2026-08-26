"use client";

import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import { Region } from "@/types/assessment";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw, Eye } from "lucide-react";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface Props {
  fileUrl: string;
  isPdf: boolean;
  regions: Region[]; // regions to highlight for the selected question
  pageSizes: number[][]; // original [width, height] per page, matching bbox coordinate space
  activePage: number;
  onPageChange: (page: number) => void;
  totalPages: number;
}

export default function AnswerSheetViewer({
  fileUrl,
  isPdf,
  regions,
  pageSizes,
  activePage,
  onPageChange,
  totalPages,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderedSize, setRenderedSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState<number>(1);

  // Jump to the first region's page whenever the selected question changes.
  useEffect(() => {
    if (regions.length > 0) {
      onPageChange(regions[0].page);
    }
  }, [regions, onPageChange]);

  const original = pageSizes[activePage - 1];
  const scaleX = original && renderedSize ? renderedSize.w / original[0] : 1;
  const scaleY = original && renderedSize ? renderedSize.h / original[1] : 1;
  const regionsOnPage = regions.filter((r) => r.page === activePage);

  const baseWidth = 640;
  const currentWidth = baseWidth * zoom;

  return (
    <div className="flex flex-col h-full bg-[#eef0f4]">
      {/* Control Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-white border-b border-slate-200 shadow-xs z-10 text-xs font-semibold text-slate-700">
        <div className="flex items-center gap-2">
          <div className="h-6 w-6 rounded-lg bg-[#fff0e8] text-[#ff5a1f] flex items-center justify-center">
            <Eye size={14} />
          </div>
          <span>Student Answer Sheet</span>
          {regionsOnPage.length > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-[#ff5a1f]/10 text-[#ff5a1f] text-[11px] font-bold">
              {regionsOnPage.length} region{regionsOnPage.length > 1 ? "s" : ""} mapped
            </span>
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
              {activePage} / {totalPages}
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
      <div className="flex-1 overflow-auto scrollbar-thin flex justify-center py-6 px-4">
        <div ref={containerRef} className="relative inline-block shadow-2xl rounded-xl overflow-hidden bg-white border border-slate-300">
          {isPdf ? (
            <Document
              file={fileUrl}
              loading={
                <div className="p-16 text-center text-xs font-medium text-slate-400 flex flex-col items-center gap-2">
                  <div className="h-6 w-6 border-2 border-[#ff5a1f] border-t-transparent rounded-full animate-spin" />
                  Loading document page...
                </div>
              }
            >
              <Page
                pageNumber={activePage}
                width={currentWidth}
                onRenderSuccess={(page) => setRenderedSize({ w: page.width, h: page.height })}
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
                setRenderedSize({ w: img.clientWidth, h: img.clientWidth * (img.naturalHeight / img.naturalWidth) });
              }}
            />
          )}

          {/* Region Bounding Box Highlights */}
          {renderedSize &&
            regionsOnPage.map((r, i) => (
              <div
                key={i}
                className="absolute border-2 border-[#ff5a1f] bg-[#ff5a1f]/20 rounded-md transition-all duration-300 shadow-[0_0_12px_rgba(255,90,31,0.5)] animate-pulse"
                style={{
                  left: r.bbox.x * scaleX,
                  top: r.bbox.y * scaleY,
                  width: r.bbox.width * scaleX,
                  height: r.bbox.height * scaleY,
                }}
              >
                <div className="absolute -top-6 left-0 bg-[#ff5a1f] text-white text-[10px] font-bold px-2 py-0.5 rounded shadow-sm flex items-center gap-1">
                  <span>Answer Region</span>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
