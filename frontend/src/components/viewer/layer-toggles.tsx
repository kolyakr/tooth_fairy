"use client";

import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

export function LayerToggles({
  layers,
  onChange,
}: {
  layers: { quadrants: boolean; teeth: boolean; periapical: boolean; caries: boolean; impacted: boolean };
  onChange: (next: {
    quadrants: boolean;
    teeth: boolean;
    periapical: boolean;
    caries: boolean;
    impacted: boolean;
  }) => void;
}) {
  return (
    <div className="space-y-3 rounded-lg border border-border bg-card p-3">
      <div className="text-sm font-medium text-foreground">Layers</div>
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="layer-q" className="text-xs">
          Quadrants
        </Label>
        <Switch id="layer-q" checked={layers.quadrants} onCheckedChange={(v) => onChange({ ...layers, quadrants: v })} />
      </div>
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="layer-t" className="text-xs">
          Teeth numbering
        </Label>
        <Switch id="layer-t" checked={layers.teeth} onCheckedChange={(v) => onChange({ ...layers, teeth: v })} />
      </div>
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="layer-p" className="text-xs">
          Periapical
        </Label>
        <Switch id="layer-p" checked={layers.periapical} onCheckedChange={(v) => onChange({ ...layers, periapical: v })} />
      </div>
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="layer-caries" className="text-xs">
          Caries
        </Label>
        <Switch id="layer-caries" checked={layers.caries} onCheckedChange={(v) => onChange({ ...layers, caries: v })} />
      </div>
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="layer-impacted" className="text-xs">
          Impacted
        </Label>
        <Switch
          id="layer-impacted"
          checked={layers.impacted}
          onCheckedChange={(v) => onChange({ ...layers, impacted: v })}
        />
      </div>
    </div>
  );
}
