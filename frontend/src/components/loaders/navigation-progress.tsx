"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { Progress } from "@/components/ui/progress";

/** Subtle top progress bar on route changes. */
export function NavigationProgress() {
  const pathname = usePathname();
  const [active, setActive] = useState(false);

  useEffect(() => {
    setActive(true);
    const t = window.setTimeout(() => setActive(false), 450);
    return () => window.clearTimeout(t);
  }, [pathname]);

  if (!active) return null;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-[100] h-0.5 overflow-hidden">
      <Progress indeterminate className="h-0.5 rounded-none border-0 bg-transparent" />
    </div>
  );
}
