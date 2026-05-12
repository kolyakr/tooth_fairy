"use client";

import * as React from "react";

import { cn } from "@/lib/cn";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  indeterminate?: boolean;
}

function Progress({ className, value = 0, indeterminate, ...props }: ProgressProps) {
  const pct = indeterminate ? undefined : Math.min(100, Math.max(0, value));

  return (
    <div className={cn("relative h-2 w-full overflow-hidden rounded-full bg-secondary", className)} {...props}>
      <div
        className={cn(
          "h-full bg-primary transition-all duration-300 ease-out",
          indeterminate && "absolute left-0 top-0 w-1/3 animate-progress-slide",
        )}
        style={
          indeterminate
            ? undefined
            : {
                width: `${pct}%`,
              }
        }
      />
    </div>
  );
}

export { Progress };
