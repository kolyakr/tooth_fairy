"use client";

import { Save } from "lucide-react";
import { memo } from "react";

import { Input } from "@/components/ui/input";

export const StatusBar = memo(function StatusBar({
  pointerPx,
  zoomPercent,
  saveState,
  reviewer,
  onReviewerChange,
  selectedSummary,
}: {
  pointerPx: { x: number; y: number } | null;
  zoomPercent: number;
  saveState: "idle" | "saving" | "saved" | "error";
  reviewer: string;
  onReviewerChange: (v: string) => void;
  selectedSummary: string | null;
}) {
  const saveLabel =
    saveState === "saving" ? "Saving…" : saveState === "saved" ? "Saved" : saveState === "error" ? "Save failed" : "Idle";

  return (
    <footer className="flex flex-wrap items-center gap-3 border-t border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
      <span className="font-mono tabular-nums">
        {pointerPx ? `${Math.round(pointerPx.x)}, ${Math.round(pointerPx.y)} px` : "— px"}
      </span>
      <span className="hidden sm:inline">·</span>
      <span>Zoom {zoomPercent}%</span>
      <span className="hidden sm:inline">·</span>
      <span className="flex items-center gap-1">
        <Save className="size-3.5 opacity-70" />
        {saveLabel}
      </span>
      <span className="hidden md:inline">·</span>
      <label className="flex items-center gap-2">
        <span className="text-muted-foreground">Reviewer</span>
        <Input value={reviewer} onChange={(e) => onReviewerChange(e.target.value)} className="h-8 max-w-[200px] text-xs" />
      </label>
      {selectedSummary ? (
        <>
          <span className="hidden lg:inline">·</span>
          <span className="max-w-[min(40vw,320px)] truncate text-foreground">{selectedSummary}</span>
        </>
      ) : null}
    </footer>
  );
});
