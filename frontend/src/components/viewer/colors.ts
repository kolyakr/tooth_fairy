import type { Finding } from "@/lib/mock-data";

/** Stroke/fill for findings — tuned for light & dark UI legibility on X-rays. */
export function strokeForFinding(f: Finding): string {
  const cls = f.finding;
  if (f.layer === "quadrants") return "#8b5cf6";
  if (f.layer === "periapical") return "#ef4444";
  if (cls === "Caries") return "#ca8a04";
  if (cls === "Impacted") return "#9333ea";
  if (cls === "Periapical Lesion") return "#dc2626";
  if (cls === "AI Tooth Mask" || cls.includes("FDI")) return "#0ea5e9";
  return "#38bdf8";
}

export function fillForFinding(f: Finding): string {
  return `${strokeForFinding(f)}33`;
}
