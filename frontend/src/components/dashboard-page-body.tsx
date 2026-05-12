"use client";

import Link from "next/link";

import { DashboardView } from "@/components/dashboard-view";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useDashboardAnalyses } from "@/hooks/use-dashboard-analyses";

export function DashboardPageBody() {
  const { analyses, loading } = useDashboardAnalyses();

  return (
    <div className="container-app space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="secondary" asChild>
          <Link href="/upload">Upload Workspace</Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="/viewer">Viewer hub</Link>
        </Button>
      </div>
      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-10 w-full max-w-md" />
          <Skeleton className="h-64 w-full rounded-xl" />
        </div>
      ) : (
        <DashboardView analyses={analyses} />
      )}
    </div>
  );
}
