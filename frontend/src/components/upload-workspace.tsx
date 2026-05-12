"use client";

import { Loader2, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createAnalysis } from "@/lib/api-client";
import { cn } from "@/lib/cn";
import { toast } from "@/lib/toast";

const REVIEWER_STORAGE_KEY = "toothfairy.reviewer.v1";

export function UploadWorkspace() {
  const router = useRouter();
  const [progress, setProgress] = useState<string>("Idle");
  const [isDragActive, setIsDragActive] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string>("");
  const [imageDims, setImageDims] = useState<{ w: number; h: number } | null>(null);
  const [error, setError] = useState<string>("");
  const [selectedObjectUrl, setSelectedObjectUrl] = useState<string | null>(null);
  const [patientId, setPatientId] = useState("");
  const [patientName, setPatientName] = useState("");
  const [age, setAge] = useState("");
  const [xrayDate, setXrayDate] = useState("");
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const fileRef = useRef<File | null>(null);

  const canAnalyze = useMemo(
    () =>
      Boolean(fileRef.current && imagePreview && patientId.trim().length > 0 && patientName.trim().length > 0),
    [imagePreview, patientId, patientName],
  );

  const setFile = (file: File | null) => {
    if (!file) return;
    const validTypes = ["image/png", "image/jpeg", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      setError("Unsupported file type. Please choose .jpg or .png.");
      return;
    }
    setError("");
    fileRef.current = file;
    setImageName(file.name);
    const objectUrl = URL.createObjectURL(file);
    if (selectedObjectUrl) {
      URL.revokeObjectURL(selectedObjectUrl);
    }
    setSelectedObjectUrl(objectUrl);
    setImagePreview(objectUrl);
    setImageDims(null);
  };

  const onAnalyze = async () => {
    const file = fileRef.current;
    if (!file || !imagePreview) {
      setError("Upload an image first.");
      return;
    }
    if (!patientId.trim()) {
      setError("Patient ID is required.");
      return;
    }
    if (!patientName.trim()) {
      setError("Patient name is required.");
      return;
    }

    const form = new FormData();
    form.append("patient_code", patientId.trim());
    form.append("patient_name", patientName.trim());
    if (age.trim()) form.append("age", age.trim());
    if (xrayDate.trim()) form.append("scan_date", xrayDate.trim());
    if (chiefComplaint.trim()) form.append("chief_complaint", chiefComplaint.trim());
    form.append("file", file);

    setSubmitting(true);
    setError("");
    setProgress("Uploading…");

    try {
      const res = await createAnalysis(form);
      setProgress("Starting analysis…");
      try {
        const stored = window.localStorage.getItem(REVIEWER_STORAGE_KEY);
        if (!stored?.trim()) {
          window.localStorage.setItem(REVIEWER_STORAGE_KEY, "Dr. Demo");
        }
      } catch {
        /* ignore */
      }
      router.push(`/viewer/${res.id}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Upload failed.";
      setError(msg);
      toast.error(msg);
      setProgress("Idle");
    } finally {
      setSubmitting(false);
    }
  };

  const openFilePicker = () => fileInputRef.current?.click();

  const clearFile = () => {
    if (selectedObjectUrl) {
      URL.revokeObjectURL(selectedObjectUrl);
    }
    fileRef.current = null;
    setSelectedObjectUrl(null);
    setImagePreview(null);
    setImageName("");
    setImageDims(null);
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <section className="relative space-y-6">
      {submitting ? (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-background/85 backdrop-blur-sm">
          <Loader2 className="size-12 animate-spin text-primary" aria-hidden />
          <p className="text-sm font-medium text-foreground">{progress}</p>
          <p className="max-w-sm text-center text-xs text-muted-foreground">Preparing your scan for AI-assisted review…</p>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(320px,420px)]">
        <Card className="overflow-hidden border-dashed">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Upload className="size-5 text-primary" aria-hidden />
              X-ray upload
            </CardTitle>
            <CardDescription>Required: PNG or JPEG · DICOM planned</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className={cn(
                "rounded-xl border-2 border-dashed p-6 transition-colors",
                isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 bg-muted/20",
              )}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragActive(true);
              }}
              onDragLeave={() => setIsDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragActive(false);
                const file = e.dataTransfer.files?.[0] ?? null;
                setFile(file);
              }}
            >
              <div className="flex flex-col items-center gap-3 text-center">
                <p className="text-sm font-medium text-foreground">Drag &amp; drop an OPG / panoramic image</p>
                <p className="text-xs text-muted-foreground">or choose a file from your computer</p>
                <Button type="button" onClick={openFilePicker}>
                  Choose image
                </Button>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />

            {error ? (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</div>
            ) : null}

            {imagePreview ? (
              <div className="space-y-3 rounded-lg border border-border bg-muted/30 p-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-semibold text-foreground">{imageName}</p>
                    <p className="font-mono text-xs text-muted-foreground">
                      {imageDims ? `${imageDims.w} × ${imageDims.h} px` : "Reading dimensions…"}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button type="button" variant="secondary" size="sm" onClick={openFilePicker}>
                      Replace
                    </Button>
                    <Button type="button" variant="outline" size="sm" onClick={clearFile}>
                      Remove
                    </Button>
                  </div>
                </div>
                <div className="relative aspect-[4/3] max-h-[340px] w-full overflow-hidden rounded-lg border border-border bg-background">
                  {/* eslint-disable-next-line @next/next/no-img-element -- blob preview */}
                  <img
                    src={imagePreview}
                    alt="Upload preview"
                    className="max-h-[340px] w-full object-contain"
                    onLoad={(e) => setImageDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                  />
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Patient &amp; scan metadata</CardTitle>
            <CardDescription>Required fields marked with * — used for audit trail only.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="grid gap-2">
              <Label htmlFor="patient-id">
                Patient ID <span className="text-destructive">*</span>
              </Label>
              <Input
                id="patient-id"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g. P-4021"
                autoComplete="off"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="patient-name">
                Patient name <span className="text-destructive">*</span>
              </Label>
              <Input id="patient-name" value={patientName} onChange={(e) => setPatientName(e.target.value)} placeholder="Full name" />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <Label htmlFor="age">Age</Label>
                <Input id="age" inputMode="numeric" value={age} onChange={(e) => setAge(e.target.value)} placeholder="Years" />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="xray-date">Date of X-ray</Label>
                <Input id="xray-date" type="date" value={xrayDate} onChange={(e) => setXrayDate(e.target.value)} />
              </div>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="complaint">Chief complaint</Label>
              <Input
                id="complaint"
                value={chiefComplaint}
                onChange={(e) => setChiefComplaint(e.target.value)}
                placeholder="Optional — e.g. pain UR quadrant"
              />
            </div>
            <Button type="button" className="w-full sm:w-auto" disabled={!canAnalyze || submitting} onClick={() => void onAnalyze()}>
              {submitting ? (
                <>
                  <Loader2 className="mr-2 size-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                "Analyze"
              )}
            </Button>
            <p className="text-xs text-muted-foreground">
              Status: <span className="font-medium text-foreground">{progress}</span>
            </p>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
