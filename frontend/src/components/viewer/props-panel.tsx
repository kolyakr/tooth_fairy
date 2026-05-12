"use client";

import {
  BoxSelect,
  Download,
  FileText,
  Hand,
  Loader2,
  MousePointer2,
  Pencil,
  Trash2,
  Undo2,
  Redo2,
} from "lucide-react";
import { memo } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { ReportPreviewResponse } from "@/lib/api-types";
import type { Finding } from "@/lib/mock-data";

import type { ViewerTool } from "./types";

export const PropsPanel = memo(function PropsPanel({
  tool,
  onTool,
  threshold,
  onThreshold,
  findingOptions,
  selectedFinding,
  onReclassify,
  onChangeToothLabel,
  onDelete,
  onUndo,
  onRedo,
  undoDisabled,
  redoDisabled,
  onMarkReviewComplete,
  onExportAudit,
  onReportFieldChange,
  onPreviewReport,
  onGenerateReport,
  onDownloadReport,
  reviewError,
  pdfDisabled,
  reportOpen,
  onReportOpenChange,
  reportDraft,
  reportPreview,
  reportBusy,
  reportError,
  reportDownloadEnabled,
  precisionMode,
  onPrecisionMode,
}: {
  tool: ViewerTool;
  onTool: (t: ViewerTool) => void;
  threshold: number;
  onThreshold: (v: number) => void;
  findingOptions: string[];
  selectedFinding: Finding | null;
  onReclassify: (label: string) => void;
  onChangeToothLabel: (label: string) => void;
  onDelete: () => void;
  onUndo: () => void;
  onRedo: () => void;
  undoDisabled: boolean;
  redoDisabled: boolean;
  onMarkReviewComplete: () => void;
  onExportAudit: () => void;
  onReportFieldChange: (
    field: "clinical_summary" | "impression" | "recommendations" | "reviewer_confirmation",
    value: string,
  ) => void;
  onPreviewReport: () => void;
  onGenerateReport: () => void;
  onDownloadReport: () => void;
  reviewError: string;
  pdfDisabled: boolean;
  reportOpen: boolean;
  onReportOpenChange: (open: boolean) => void;
  reportDraft: {
    clinical_summary: string;
    impression: string;
    recommendations: string;
    reviewer_confirmation: string;
  };
  reportPreview: ReportPreviewResponse | null;
  reportBusy: boolean;
  reportError: string;
  reportDownloadEnabled: boolean;
  precisionMode: boolean;
  onPrecisionMode: (v: boolean) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-border bg-card p-3">
        <div className="mb-2 text-sm font-medium text-foreground">Tools</div>
        <ToggleGroup
          type="single"
          value={tool}
          onValueChange={(v) => v && onTool(v as ViewerTool)}
          className="grid grid-cols-2 gap-2"
        >
          <ToggleGroupItem value="select" variant="outline" aria-label="Select" className="gap-2">
            <MousePointer2 className="size-4" />
            Select
          </ToggleGroupItem>
          <ToggleGroupItem value="pan" variant="outline" aria-label="Pan" className="gap-2">
            <Hand className="size-4" />
            Pan
          </ToggleGroupItem>
          <ToggleGroupItem value="drawPolygon" variant="outline" aria-label="Polygon" className="gap-2">
            <Pencil className="size-4" />
            Polygon
          </ToggleGroupItem>
          <ToggleGroupItem value="drawBox" variant="outline" aria-label="Box" className="gap-2">
            <BoxSelect className="size-4" />
            Box
          </ToggleGroupItem>
          <ToggleGroupItem value="erase" variant="outline" aria-label="Erase" className="col-span-2 gap-2">
            <Trash2 className="size-4" />
            Erase (click shape)
          </ToggleGroupItem>
        </ToggleGroup>
        <label className="mt-3 flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
          <input type="checkbox" checked={precisionMode} onChange={(e) => onPrecisionMode(e.target.checked)} />
          Precision drag (Shift also works)
        </label>
      </div>

      <div className="rounded-lg border border-border bg-card p-3">
        <Label className="text-xs text-muted-foreground">Confidence filter</Label>
        <Slider className="mt-2" min={0} max={1} step={0.05} value={[threshold]} onValueChange={(v) => onThreshold(v[0] ?? 0)} />
        <p className="mt-1 text-[10px] text-muted-foreground">Hide AI suggestions below {Math.round(threshold * 100)}%</p>
      </div>

      <div className="rounded-lg border border-border bg-card p-3">
        <div className="text-sm font-medium text-foreground">Selected finding</div>
        {!selectedFinding ? (
          <p className="mt-2 text-sm text-muted-foreground">Select a polygon on the canvas or from the list.</p>
        ) : (
          <div className="mt-3 grid gap-3">
            <div className="grid gap-1">
              <Label className="text-xs">Classification</Label>
              <Select value={selectedFinding.finding} onValueChange={onReclassify}>
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {findingOptions.map((opt) => (
                    <SelectItem key={opt} value={opt}>
                      {opt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-1">
              <Label className="text-xs">Tooth label</Label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                value={selectedFinding.toothLabel}
                onChange={(e) => onChangeToothLabel(e.target.value)}
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" type="button" variant="secondary" onClick={() => onDelete()}>
                Delete
              </Button>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        <Button variant="outline" size="sm" type="button" disabled={undoDisabled} onClick={onUndo} className="gap-1">
          <Undo2 className="size-3.5" />
          Undo
        </Button>
        <Button variant="outline" size="sm" type="button" disabled={redoDisabled} onClick={onRedo} className="gap-1">
          <Redo2 className="size-3.5" />
          Redo
        </Button>
      </div>

      <div className="rounded-lg border border-border bg-card p-3">
        <Button className="w-full" type="button" onClick={onMarkReviewComplete}>
          Mark review complete
        </Button>
        {reviewError ? <p className="mt-2 text-xs text-destructive">{reviewError}</p> : null}
        <Button variant="secondary" className="mt-2 w-full" type="button" onClick={onExportAudit}>
          Export audit JSON
        </Button>
        <Dialog open={reportOpen} onOpenChange={onReportOpenChange}>
          <DialogTrigger asChild>
            <Button variant="outline" className="mt-2 w-full gap-2" type="button">
              <FileText className="size-4" />
              Report workspace
            </Button>
          </DialogTrigger>
          <DialogContent className="max-h-[85vh] max-w-4xl overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Generate PDF report</DialogTitle>
              <DialogDescription>
                Edit clinical narrative, preview structured content, then generate the final PDF.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-3">
                <div className="grid gap-1.5">
                  <Label>Clinical summary</Label>
                  <textarea
                    className="min-h-24 rounded-md border border-input bg-transparent px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    value={reportDraft.clinical_summary}
                    onChange={(e) => onReportFieldChange("clinical_summary", e.target.value)}
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label>Impression</Label>
                  <textarea
                    className="min-h-20 rounded-md border border-input bg-transparent px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    value={reportDraft.impression}
                    onChange={(e) => onReportFieldChange("impression", e.target.value)}
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label>Recommendations</Label>
                  <textarea
                    className="min-h-24 rounded-md border border-input bg-transparent px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    value={reportDraft.recommendations}
                    onChange={(e) => onReportFieldChange("recommendations", e.target.value)}
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label>Reviewer confirmation</Label>
                  <Input
                    value={reportDraft.reviewer_confirmation}
                    onChange={(e) => onReportFieldChange("reviewer_confirmation", e.target.value)}
                    placeholder="I reviewed and confirm this report."
                  />
                </div>
              </div>
              <div className="space-y-3 rounded-lg border border-border bg-muted/20 p-3">
                <div className="text-sm font-medium">Preview</div>
                {!reportPreview ? (
                  <p className="text-sm text-muted-foreground">
                    Click Preview to build report data from current draft and accepted findings.
                  </p>
                ) : (
                  <div className="space-y-3 text-sm">
                    <div className="rounded-md border border-border bg-card p-2">
                      <p className="font-medium">{reportPreview.patient_name}</p>
                      <p className="text-xs text-muted-foreground">
                        {reportPreview.patient_code} · accepted findings: {reportPreview.accepted_findings_count}
                      </p>
                    </div>
                    {reportPreview.sections.map((section) => (
                      <div key={section.title} className="rounded-md border border-border bg-card p-2">
                        <p className="text-xs font-semibold text-muted-foreground">{section.title}</p>
                        <p className="mt-1 text-sm">{section.body}</p>
                      </div>
                    ))}
                  </div>
                )}
                {reportError ? <p className="text-xs text-destructive">{reportError}</p> : null}
              </div>
            </div>
            <DialogFooter className="gap-2">
              <Button type="button" variant="secondary" onClick={onPreviewReport} disabled={reportBusy}>
                {reportBusy ? <Loader2 className="mr-2 size-4 animate-spin" /> : null}
                Preview
              </Button>
              <Button type="button" onClick={onGenerateReport} disabled={pdfDisabled || reportBusy}>
                {reportBusy ? <Loader2 className="mr-2 size-4 animate-spin" /> : null}
                {pdfDisabled ? "Generate PDF (after review)" : "Generate PDF"}
              </Button>
              <Button
                type="button"
                variant="outline"
                className="gap-2"
                onClick={onDownloadReport}
                disabled={!reportDownloadEnabled || reportBusy}
              >
                <Download className="size-4" />
                Download PDF
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
});
