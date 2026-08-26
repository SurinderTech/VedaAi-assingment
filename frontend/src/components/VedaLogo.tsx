import React from "react";

interface VedaLogoProps {
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
  showText?: boolean;
}

export default function VedaLogo({ size = "md", className = "", showText = true }: VedaLogoProps) {
  const sizes = {
    sm: { box: "h-8 w-8 rounded-xl", text: "text-lg", icon: "w-5 h-5" },
    md: { box: "h-10 w-10 rounded-2xl", text: "text-xl", icon: "w-6 h-6" },
    lg: { box: "h-12 w-12 rounded-2xl", text: "text-2xl", icon: "w-7 h-7" },
    xl: { box: "h-16 w-16 rounded-3xl", text: "text-3xl", icon: "w-10 h-10" },
  };

  const s = sizes[size];

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Black rounded container with ultra-thick bold white V */}
      <div className={`${s.box} bg-[#1a1a1a] flex items-center justify-center text-white shadow-sm shrink-0`}>
        <svg
          className={`${s.icon} fill-white`}
          viewBox="0 0 24 24"
        >
          {/* Ultra-thick chunky white V logo */}
          <path d="M1 2.5h7.5l3.5 12.8L15.5 2.5H23L15.2 21.5h-6.4L1 2.5z" />
        </svg>
      </div>

      {showText && (
        <span className={`${s.text} font-extrabold text-[#1a1a1a] tracking-tight`}>
          VedaAI
        </span>
      )}
    </div>
  );
}
