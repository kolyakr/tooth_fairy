"use client";

import Link from "next/link";
import { ArrowDownAZ, ArrowUpAZ, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AnalysisListItem, AnalysisStatus, AlertLevel } from "@/lib/api-types";
import { cn } from "@/lib/cn";

const STATUS_TAB_ALL = "all";

function statusTone(status: AnalysisStatus): string {
  switch (status) {
    case "Pending AI":
      return "border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-100";
    case "Reviewing":
      return "border-sky-500/40 bg-sky-500/10 text-sky-900 dark:text-sky-100";
    case "Reviewed":
      return "border-emerald-500/40 bg-emerald-500/10 text-emerald-900 dark:text-emerald-100";
    case "Report Generated":
      return "border-violet-500/40 bg-violet-500/10 text-violet-900 dark:text-violet-100";
    case "Failed":
      return "border-destructive/50 bg-destructive/10 text-destructive";
    default:
      return "";
  }
}

function alertTone(level: AlertLevel | null): string {
  switch (level) {
    case "High":
      return "border-red-500/40 bg-red-500/10 text-red-800 dark:text-red-200";
    case "Medium":
      return "border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-100";
    case "Low":
    default:
      return "border-emerald-500/40 bg-emerald-500/10 text-emerald-900 dark:text-emerald-100";
  }
}

export function DashboardView({ analyses }: { analyses: AnalysisListItem[] }) {
  const [query, setQuery] = useState("");
  const [statusTab, setStatusTab] = useState<string>(STATUS_TAB_ALL);
  const [sortDesc, setSortDesc] = useState(true);

  const stats = useMemo(() => {
    const total = analyses.length;
    const pending = analyses.filter((a) => a.status === "Pending AI").length;
    const reviewing = analyses.filter((a) => a.status === "Reviewing").length;
    const reviewed = analyses.filter((a) => a.status === "Reviewed" || a.status === "Report Generated").length;
    return { total, pending, reviewing, reviewed };
  }, [analyses]);

  const filtered = useMemo(() => {
    let rows = analyses.filter((a) => {
      const q = query.trim().toLowerCase();
      const matchesQuery =
        !q ||
        a.patient_name.toLowerCase().includes(q) ||
        a.patient_id.toLowerCase().includes(q);
      const matchesStatus = statusTab === STATUS_TAB_ALL || a.status === statusTab;
      return matchesQuery && matchesStatus;
    });
    rows = [...rows].sort((a, b) => {
      const da = a.scan_date ? new Date(a.scan_date).getTime() : 0;
      const db = b.scan_date ? new Date(b.scan_date).getTime() : 0;
      return sortDesc ? db - da : da - db;
    });
    return rows;
  }, [analyses, query, sortDesc, statusTab]);

  return (
    <div className="space-y-8">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card className="border-primary/20 bg-gradient-to-br from-card to-primary/5">
          <CardHeader className="pb-2">
            <CardDescription>Total analyses</CardDescription>
            <CardTitle className="text-3xl tabular-nums">{stats.total}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-amber-500/20 bg-gradient-to-br from-card to-amber-500/5">
          <CardHeader className="pb-2">
            <CardDescription>Pending AI</CardDescription>
            <CardTitle className="text-3xl tabular-nums text-amber-700 dark:text-amber-300">{stats.pending}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-sky-500/20 bg-gradient-to-br from-card to-sky-500/5">
          <CardHeader className="pb-2">
            <CardDescription>Reviewing</CardDescription>
            <CardTitle className="text-3xl tabular-nums text-sky-700 dark:text-sky-300">{stats.reviewing}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-emerald-500/20 bg-gradient-to-br from-card to-emerald-500/5">
          <CardHeader className="pb-2">
            <CardDescription>Reviewed / Report</CardDescription>
            <CardTitle className="text-3xl tabular-nums text-emerald-700 dark:text-emerald-300">{stats.reviewed}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle>Recent analyses</CardTitle>
            <CardDescription>Click a row to open the interactive viewer.</CardDescription>
          </div>
          <Button asChild>
            <Link href="/upload">New Patient Scan</Link>
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Filter by patient name or ID…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="h-10 pl-10"
              />
            </div>
            <Button variant="outline" size="sm" className="gap-2" onClick={() => setSortDesc((s) => !s)}>
              {sortDesc ? <ArrowDownAZ className="size-4" /> : <ArrowUpAZ className="size-4" />}
              Date {sortDesc ? "newest" : "oldest"}
            </Button>
          </div>

          <Tabs value={statusTab} onValueChange={setStatusTab}>
            <TabsList className="flex h-auto w-full flex-wrap justify-start gap-1">
              <TabsTrigger value={STATUS_TAB_ALL}>All</TabsTrigger>
              <TabsTrigger value="Pending AI">Pending AI</TabsTrigger>
              <TabsTrigger value="Reviewing">Reviewing</TabsTrigger>
              <TabsTrigger value="Reviewed">Reviewed</TabsTrigger>
              <TabsTrigger value="Report Generated">Report</TabsTrigger>
              <TabsTrigger value="Failed">Failed</TabsTrigger>
            </TabsList>

            <TabsContent value={statusTab} className="mt-4 outline-none">
              {filtered.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-muted/30 px-6 py-16 text-center">
                  <p className="mb-2 text-lg font-medium text-foreground">No analyses match</p>
                  <p className="mb-6 max-w-md text-sm text-muted-foreground">
                    Upload a panoramic X-ray to start an AI-assisted review pipeline.
                  </p>
                  <Button asChild>
                    <Link href="/upload">Upload scan</Link>
                  </Button>
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-border">
                  <table className="w-full min-w-[640px] text-left text-sm">
                    <thead className="border-b border-border bg-muted/50">
                      <tr>
                        <th className="px-4 py-3 font-medium">Patient</th>
                        <th className="px-4 py-3 font-medium">Scan date</th>
                        <th className="px-4 py-3 font-medium">Status</th>
                        <th className="px-4 py-3 font-medium">AI alert</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((item) => {
                        const level = item.alert_level ?? "Low";
                        return (
                          <tr
                            key={item.id}
                            className="border-b border-border/80 transition-colors last:border-0 hover:bg-muted/40"
                          >
                            <td className="px-4 py-3">
                              <Link href={`/viewer/${item.id}`} className="block rounded-md outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
                                <span className="font-semibold text-foreground">{item.patient_name}</span>
                                <span className="mt-0.5 block text-xs text-muted-foreground">{item.patient_id}</span>
                              </Link>
                            </td>
                            <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
                              {item.scan_date ? String(item.scan_date).slice(0, 10) : "—"}
                            </td>
                            <td className="px-4 py-3">
                              <Badge variant="outline" className={cn("font-normal", statusTone(item.status))}>
                                {item.status}
                              </Badge>
                            </td>
                            <td className="px-4 py-3">
                              <Badge variant="outline" className={cn("font-normal", alertTone(item.alert_level ?? null))}>
                                {level}
                              </Badge>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
