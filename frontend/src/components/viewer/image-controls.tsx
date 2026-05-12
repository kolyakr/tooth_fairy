"use client";

import { RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Toggle } from "@/components/ui/toggle";

export function ImageControls({
  brightness,
  contrast,
  gamma,
  invert,
  onBrightness,
  onContrast,
  onGamma,
  onInvert,
  onReset,
}: {
  brightness: number;
  contrast: number;
  gamma: number;
  invert: boolean;
  onBrightness: (v: number) => void;
  onContrast: (v: number) => void;
  onGamma: (v: number) => void;
  onInvert: (v: boolean) => void;
  onReset: () => void;
}) {
  return (
    <div className="space-y-3 rounded-lg border border-border bg-card p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-foreground">Image</span>
        <div className="flex items-center gap-2">
          <Toggle pressed={invert} onPressedChange={onInvert} variant="outline" size="sm" aria-label="Invert">
            Inv
          </Toggle>
          <Button variant="ghost" size="sm" type="button" className="h-8 gap-1 text-xs" onClick={onReset}>
            <RotateCcw className="size-3.5" />
            Reset
          </Button>
        </div>
      </div>
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">Brightness</Label>
        <Slider min={-0.35} max={0.35} step={0.01} value={[brightness]} onValueChange={(v) => onBrightness(v[0] ?? 0)} />
      </div>
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">Contrast</Label>
        <Slider min={-40} max={40} step={1} value={[contrast]} onValueChange={(v) => onContrast(v[0] ?? 0)} />
      </div>
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">Gamma (latitude)</Label>
        <Slider min={0.5} max={2} step={0.05} value={[gamma]} onValueChange={(v) => onGamma(v[0] ?? 1)} />
      </div>
      <p className="text-[10px] leading-snug text-muted-foreground">
        Adjust for monitor / X-ray latitude. Gamma shifts mid-tones (Konva contrast blend). Invert helps radiograph contrast preference.
      </p>
    </div>
  );
}
