"use client";

import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { getAnalysis } from "@/lib/api-client";
import type { AnalysisDetail, AnalysisStatus as AnalysisStatusValue } from "@/lib/api-types";
import { cn } from "@/lib/cn";

import { InteractiveViewer } from "./interactive-viewer";

const POLL_MS = 1500;
const STEP_LABELS = [
  "Segmenting Quadrants…",
  "Numbering Teeth (FDI)…",
  "Detecting Periapical Lesions…",
  "Classifying Teeth (Caries / Impacted)…",
];

function statusAllowsViewer(status: AnalysisStatusValue): boolean {
  return status === "Reviewing" || status === "Reviewed" || status === "Report Generated";
}

export function AnalysisStatusPanel({ analysisId }: { analysisId: string }) {
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [pollError, setPollError] = useState<string>("");
  const [visualStep, setVisualStep] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const d = await getAnalysis(analysisId);
        if (!cancelled) {
          setDetail(d);
          setPollError("");
        }
      } catch (e) {
        if (!cancelled) {
          setPollError(e instanceof Error ? e.message : "Could not load analysis.");
        }
      }
    };

    void poll();
    const id = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [analysisId]);

  const pending = detail?.status === "Pending AI";

  useEffect(() => {
    if (!pending) return;
    const id = window.setInterval(() => {
      setVisualStep((s) => Math.min(s + 1, STEP_LABELS.length - 1));
    }, 1400);
    return () => window.clearInterval(id);
  }, [pending]);

  useEffect(() => {
    if (!pending) {
      setElapsedMs(0);
      return;
    }
    const started = Date.now();
    const id = window.setInterval(() => setElapsedMs(Date.now() - started), 400);
    return () => window.clearInterval(id);
  }, [pending, analysisId]);

  useEffect(() => {
    if (detail?.status === "Pending AI") {
      setVisualStep(0);
    }
  }, [detail?.status, analysisId]);

  const activeStep = useMemo(() => {
    if (!pending) return STEP_LABELS.length;
    return Math.min(visualStep, STEP_LABELS.length - 1);
  }, [pending, visualStep]);

  if (pollError && !detail) {
    return (
      <Card className="max-w-lg border-destructive/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="size-5" aria-hidden />
            Could not load analysis
          </CardTitle>
          <CardDescription>{pollError}</CardDescription>
        </CardHeader>
        <CardFooter>
          <Button asChild>
            <Link href="/upload">Retry upload</Link>
          </Button>
        </CardFooter>
      </Card>
    );
  }

  if (detail?.status === "Failed") {
    return (
      <Card className="max-w-lg border-destructive/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="size-5 text-destructive" aria-hidden />
            Analysis failed
          </CardTitle>
          <CardDescription>{detail.error ?? "Inference failed. Please try another image."}</CardDescription>
        </CardHeader>
        <CardFooter className="flex flex-wrap gap-2">
          <Button asChild variant="default">
            <Link href="/upload">Retry upload</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/">Dashboard</Link>
          </Button>
        </CardFooter>
      </Card>
    );
  }

  if (detail && statusAllowsViewer(detail.status)) {
    return <InteractiveViewer analysisId={analysisId} initialDetail={detail} />;
  }

  return (
    <Card className="max-w-xl overflow-hidden">
      <CardHeader className="space-y-3">
        <div className="space-y-1">
          <CardTitle>Running AI pipeline</CardTitle>
          <CardDescription>
            Analysis <span className="font-mono text-foreground">{analysisId}</span>
          </CardDescription>
        </div>
        <Progress indeterminate className="h-1.5" />
        <p className="text-xs text-muted-foreground">
          Elapsed {(elapsedMs / 1000).toFixed(1)}s · indeterminate until server marks review-ready
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <ol className="grid gap-3">
          {STEP_LABELS.map((label, idx) => {
            const done = idx < activeStep;
            const active = idx === activeStep && pending;
            return (
              <li
                key={label}
                className={cn(
                  "flex items-start gap-3 rounded-lg border px-3 py-2.5 text-sm transition-colors",
                  done ? "border-primary/30 bg-primary/5 text-foreground" : active ? "border-primary bg-primary/10 shadow-sm" : "border-border/80 bg-muted/20 text-muted-foreground",
                )}
              >
                <span className="mt-0.5 shrink-0">
                  {done ? (
                    <CheckCircle2 className="size-5 text-primary" aria-hidden />
                  ) : active ? (
                    <Loader2 className="size-5 animate-spin text-primary" aria-hidden />
                  ) : (
                    <span className="flex size-5 items-center justify-center rounded-full border border-muted-foreground/40 text-[10px] font-semibold">
                      {idx + 1}
                    </span>
                  )}
                </span>
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className={cn("font-medium", active && "text-primary")}>{label}</span>
                  {active ? <span className="text-xs text-primary/90">Active step — simulated pacing for UX</span> : null}
                </span>
              </li>
            );
          })}
        </ol>
        <p className="text-sm text-muted-foreground">
          Waiting for server status… This page refreshes automatically every few seconds.
        </p>
        {pollError ? <p className="text-sm text-amber-700 dark:text-amber-400">{pollError}</p> : null}
      </CardContent>
    </Card>
  );
}
