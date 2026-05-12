"use client";

import type { ImageAssetKind } from "@/lib/api-types";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

const OVERLAYS: { value: ImageAssetKind | "original"; label: string }[] = [
  { value: "original", label: "Original" },
  { value: "quadrants_overlay", label: "Quadrants" },
  { value: "teeth_overlay", label: "Teeth" },
  { value: "periapical_quadrants_overlay", label: "Periapical" },
  { value: "teeth_classification_overlay", label: "Classification" },
];

export function OverlayTabs({
  value,
  onChange,
}: {
  value: ImageAssetKind | "original";
  onChange: (v: ImageAssetKind | "original") => void;
}) {
  return (
    <Tabs value={value} onValueChange={(v) => onChange(v as ImageAssetKind | "original")} className="w-full">
      <TabsList className="flex h-auto min-h-9 w-full flex-wrap justify-start gap-1 bg-muted/60 p-1">
        {OVERLAYS.map((o) => (
          <TabsTrigger key={o.value} value={o.value} className="text-xs">
            {o.label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
