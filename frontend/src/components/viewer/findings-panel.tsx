"use client";

import { Check, ChevronDown, ChevronRight, Search, X } from "lucide-react";
import { memo, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { Finding } from "@/lib/mock-data";
import { cn } from "@/lib/cn";

import { strokeForFinding } from "./colors";

function groupByTooth(rows: Finding[]) {
  const m = new Map<string, Finding[]>();
  rows.forEach((f) => {
    const k = f.toothLabel;
    m.set(k, [...(m.get(k) ?? []), f]);
  });
  return Array.from(m.entries()).sort(([a], [b]) => a.localeCompare(b));
}

function groupByClass(rows: Finding[]) {
  const m = new Map<string, Finding[]>();
  rows.forEach((f) => {
    const k = f.finding;
    m.set(k, [...(m.get(k) ?? []), f]);
  });
  return Array.from(m.entries()).sort(([a], [b]) => a.localeCompare(b));
}

export const FindingsPanel = memo(function FindingsPanel({
  visibleRows,
  selectedId,
  onSelect,
  onAccept,
  onReject,
  onHover,
  onCenter,
  onReclassify,
  onDelete,
  reclassifyOptions,
  actionStateById,
}: {
  visibleRows: Finding[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAccept: (id: string) => void;
  onReject: (id: string) => void;
  onHover: (id: string | null) => void;
  onCenter: (id: string) => void;
  onReclassify: (id: string, label: string) => void;
  onDelete: (id: string) => void;
  reclassifyOptions: string[];
  actionStateById: Record<string, "idle" | "saving" | "error">;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return visibleRows;
    return visibleRows.filter(
      (f) =>
        f.toothLabel.toLowerCase().includes(s) ||
        f.finding.toLowerCase().includes(s) ||
        f.id.toLowerCase().includes(s),
    );
  }, [q, visibleRows]);

  const pending = useMemo(() => filtered.filter((f) => !f.accepted), [filtered]);

  const byTooth = useMemo(() => groupByTooth(filtered), [filtered]);
  const byClass = useMemo(() => groupByClass(filtered), [filtered]);

  const Row = ({ f }: { f: Finding }) => (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div
          role="button"
          tabIndex={0}
          className={cn(
            "flex cursor-pointer items-center gap-2 rounded-lg border px-2 py-2 text-left text-sm transition-colors",
            selectedId === f.id ? "border-primary bg-primary/10" : "border-transparent bg-muted/30 hover:bg-muted/60",
          )}
          onClick={() => onSelect(f.id)}
          onMouseEnter={() => onHover(f.id)}
          onMouseLeave={() => onHover(null)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSelect(f.id);
            }
          }}
        >
          <span className="h-3 w-3 shrink-0 rounded-full border border-white/30 shadow-sm" style={{ backgroundColor: strokeForFinding(f) }} />
          <div className="min-w-0 flex-1">
            <div className="truncate font-medium text-foreground">{f.finding}</div>
            <div className="truncate text-xs text-muted-foreground">
              {f.toothLabel} · {Math.round(f.confidence * 100)}%
            </div>
            <div className="mt-1">
              <Badge variant={f.accepted ? "secondary" : "outline"} className="text-[10px]">
                {f.accepted ? "Approved" : "Declined"}
              </Badge>
              {actionStateById[f.id] === "saving" ? (
                <Badge variant="outline" className="ml-1 text-[10px]">
                  Saving...
                </Badge>
              ) : null}
              {actionStateById[f.id] === "error" ? (
                <Badge variant="destructive" className="ml-1 text-[10px]">
                  Failed
                </Badge>
              ) : null}
            </div>
          </div>
          <div className="flex shrink-0 gap-0.5">
            <Button
              size="icon"
              variant="ghost"
              className="size-8 text-emerald-600 hover:bg-emerald-500/15 hover:text-emerald-700"
              type="button"
              aria-label="Accept"
              disabled={f.accepted || actionStateById[f.id] === "saving"}
              onClick={(e) => {
                e.stopPropagation();
                onAccept(f.id);
              }}
            >
              <Check className="size-4" />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              className="size-8 text-destructive hover:bg-destructive/10"
              type="button"
              aria-label="Reject"
              disabled={!f.accepted || actionStateById[f.id] === "saving"}
              onClick={(e) => {
                e.stopPropagation();
                onReject(f.id);
              }}
            >
              <X className="size-4" />
            </Button>
          </div>
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => onCenter(f.id)}>Center</ContextMenuItem>
        <ContextMenuSub>
          <ContextMenuSubTrigger>Reclassify</ContextMenuSubTrigger>
          <ContextMenuSubContent>
            {reclassifyOptions.map((opt) => (
              <ContextMenuItem key={opt} onSelect={() => onReclassify(f.id, opt)}>
                {opt}
              </ContextMenuItem>
            ))}
          </ContextMenuSubContent>
        </ContextMenuSub>
        <ContextMenuSeparator />
        <ContextMenuItem className="text-destructive focus:text-destructive" onSelect={() => onDelete(f.id)}>
          Delete
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col rounded-xl border border-border bg-card shadow-sm">
      <div className="border-b border-border p-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input placeholder="Search findings…" value={q} onChange={(e) => setQ(e.target.value)} className="h-9 pl-9 text-sm" />
        </div>
      </div>
      <Tabs defaultValue="tooth" className="flex min-h-0 flex-1 flex-col px-2 pt-2">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="tooth" className="text-xs">
            By tooth
          </TabsTrigger>
          <TabsTrigger value="class" className="text-xs">
            By class
          </TabsTrigger>
          <TabsTrigger value="pending" className="text-xs">
            Pending
          </TabsTrigger>
        </TabsList>
        <TabsContent value="tooth" className="mt-2 min-h-0 flex-1 overflow-hidden">
          <ScrollArea className="h-full pr-3">
            <div className="space-y-3 pb-4">
              {byTooth.map(([tooth, items]) => {
                const isOpen = open[tooth] ?? true;
                return (
                  <div key={tooth}>
                    <button
                      type="button"
                      className="mb-1 flex w-full items-center gap-1 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                      onClick={() => setOpen((o) => ({ ...o, [tooth]: !isOpen }))}
                    >
                      {isOpen ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
                      {tooth}
                      <Badge variant="secondary" className="ml-auto font-mono text-[10px]">
                        {items.length}
                      </Badge>
                    </button>
                    {isOpen ? (
                      <div className="space-y-1.5 pl-1">
                        {items.map((f) => (
                          <Row key={f.id} f={f} />
                        ))}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        </TabsContent>
        <TabsContent value="class" className="mt-2 min-h-0 flex-1 overflow-hidden">
          <ScrollArea className="h-full pr-3">
            <div className="space-y-3 pb-4">
              {byClass.map(([cls, items]) => {
                const key = `c:${cls}`;
                const isOpen = open[key] ?? true;
                return (
                  <div key={cls}>
                    <button
                      type="button"
                      className="mb-1 flex w-full items-center gap-1 text-left text-xs font-semibold text-muted-foreground"
                      onClick={() => setOpen((o) => ({ ...o, [key]: !isOpen }))}
                    >
                      {isOpen ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
                      {cls}
                      <Badge variant="outline" className="ml-auto text-[10px]">
                        {items.length}
                      </Badge>
                    </button>
                    {isOpen ? (
                      <div className="space-y-1.5 pl-1">
                        {items.map((f) => (
                          <Row key={f.id} f={f} />
                        ))}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        </TabsContent>
        <TabsContent value="pending" className="mt-2 min-h-0 flex-1 overflow-hidden">
          <ScrollArea className="h-full pr-3">
            <div className="space-y-1.5 pb-4">
              {pending.length === 0 ? (
                <p className="py-8 text-center text-sm text-muted-foreground">No pending findings.</p>
              ) : (
                pending.map((f) => <Row key={f.id} f={f} />)
              )}
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
});
