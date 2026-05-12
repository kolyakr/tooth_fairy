"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";
import { ArrowRight, Clock3, FolderOpen, ScanLine } from "lucide-react";

import { TopNav } from "@/components/top-nav";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useDashboardAnalyses } from "@/hooks/use-dashboard-analyses";

/** Legacy query `?id=` redirects to `/viewer/[id]`. List loads with Bearer when signed in. */
export function ViewerHub() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { analyses, loading } = useDashboardAnalyses(8);

  useEffect(() => {
    const id = searchParams.get("id");
    if (id?.trim()) {
      router.replace(`/viewer/${encodeURIComponent(id.trim())}`);
    }
  }, [router, searchParams]);

  return (
    <main className="min-h-screen bg-background">
      <TopNav />
      <div className="container-app space-y-6">
        <Card className="border-primary/20 bg-gradient-to-b from-primary/5 to-card">
          <CardHeader className="space-y-3">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1 text-xs text-muted-foreground">
              <FolderOpen className="size-3.5" />
              Viewer hub
            </div>
            <CardTitle className="text-2xl">No analysis selected yet</CardTitle>
            <CardDescription className="max-w-2xl text-sm">
              Open an existing analysis from the list below or start a new upload. Once an analysis is selected, you will
              enter the full interactive viewer with editing tools and audit tracking.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Button asChild>
              <Link href="/upload">
                <ScanLine className="size-4" />
                New Patient Scan
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/">Back to Dashboard</Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Recent analyses</CardTitle>
            <CardDescription>Jump directly into review for recently processed scans.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {loading ? (
              <div className="space-y-2">
                <Skeleton className="h-14 w-full rounded-lg" />
                <Skeleton className="h-14 w-full rounded-lg" />
              </div>
            ) : analyses.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border px-4 py-6 text-sm text-muted-foreground">
                No analyses found yet. Upload a scan to create your first case, or sign in to see persisted analyses.
              </div>
            ) : (
              analyses.map((row) => (
                <Link
                  key={row.id}
                  href={`/viewer/${row.id}`}
                  className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-3 py-2.5 transition hover:bg-muted/40"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-foreground">{row.patient_name}</p>
                    <p className="truncate text-xs text-muted-foreground">
                      {row.patient_id} · {row.status}
                    </p>
                  </div>
                  <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock3 className="size-3.5" />
                    Open
                    <ArrowRight className="size-3.5" />
                  </span>
                </Link>
              ))
            )}
          </CardContent>
        </Card>
        <div className="text-xs text-muted-foreground">
          Tip: You can also open a specific analysis directly using `/viewer/{`id`}` deep links.
        </div>
      </div>
    </main>
  );
}
