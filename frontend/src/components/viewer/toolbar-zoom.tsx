"use client";

import { Focus, Maximize2, Minus, Plus, Scan } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function ToolbarZoom({
  zoomPercent,
  onZoomIn,
  onZoomOut,
  onFit,
  onActual,
  onFitSelected,
  disabledFitSelected,
}: {
  zoomPercent: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onActual: () => void;
  onFitSelected: () => void;
  disabledFitSelected: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-card/80 px-2 py-1.5 shadow-sm">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="outline" size="icon" className="size-8" type="button" onClick={onZoomOut} aria-label="Zoom out">
            <Minus className="size-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Zoom out (−)</TooltipContent>
      </Tooltip>
      <span className="min-w-[4rem] text-center font-mono text-xs tabular-nums text-muted-foreground">{zoomPercent}%</span>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="outline" size="icon" className="size-8" type="button" onClick={onZoomIn} aria-label="Zoom in">
            <Plus className="size-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Zoom in (+)</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="secondary" size="sm" className="gap-1.5" type="button" onClick={onFit}>
            <Maximize2 className="size-3.5" />
            Fit
          </Button>
        </TooltipTrigger>
        <TooltipContent>Fit to window (0)</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="secondary" size="sm" className="gap-1.5" type="button" onClick={onActual}>
            <Scan className="size-3.5" />
            1:1
          </Button>
        </TooltipTrigger>
        <TooltipContent>Actual pixels (1)</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="secondary" size="sm" className="gap-1.5" type="button" onClick={onFitSelected} disabled={disabledFitSelected}>
            <Focus className="size-3.5" />
            Fit sel.
          </Button>
        </TooltipTrigger>
        <TooltipContent>Fit selected finding (F)</TooltipContent>
      </Tooltip>
    </div>
  );
}
