"use client";

import { HelpCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Kbd } from "@/components/ui/kbd";

const ROWS: { keys: string[]; desc: string }[] = [
  { keys: ["V"], desc: "Select tool" },
  { keys: ["H"], desc: "Pan tool" },
  { keys: ["P"], desc: "Draw polygon" },
  { keys: ["B"], desc: "Draw box" },
  { keys: ["E"], desc: "Erase (click polygon)" },
  { keys: ["Space"], desc: "Temporary pan while held" },
  { keys: ["+", "−"], desc: "Zoom in / out" },
  { keys: ["0"], desc: "Fit to window" },
  { keys: ["1"], desc: "Actual size (1:1)" },
  { keys: ["F"], desc: "Fit selected finding" },
  { keys: ["Shift"], desc: "Precision / axis constrain while dragging" },
  { keys: ["⌘", "Z"], desc: "Undo" },
  { keys: ["⌘", "⇧", "Z"], desc: "Redo" },
  { keys: ["A"], desc: "Accept selected finding" },
  { keys: ["R"], desc: "Reject selected finding" },
  { keys: ["Del"], desc: "Delete selected" },
  { keys: ["Esc"], desc: "Clear draft / selection" },
];

export function ShortcutsDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" type="button" className="gap-1.5">
          <HelpCircle className="size-4" />
          Shortcuts
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Keyboard shortcuts</DialogTitle>
        </DialogHeader>
        <ul className="grid gap-2 text-sm">
          {ROWS.map((row) => (
            <li key={row.desc} className="flex items-start justify-between gap-4 border-b border-border/60 py-2 last:border-0">
              <span className="text-muted-foreground">{row.desc}</span>
              <span className="flex shrink-0 flex-wrap justify-end gap-1">
                {row.keys.map((k) => (
                  <Kbd key={`${row.desc}-${k}`}>{k}</Kbd>
                ))}
              </span>
            </li>
          ))}
        </ul>
      </DialogContent>
    </Dialog>
  );
}
