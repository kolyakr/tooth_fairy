import type { FindingRead, FindingCreatePayload, FindingUpdatePayload } from "./api-types";
import type { Finding } from "./mock-data";

export function apiFindingToUi(f: FindingRead): Finding {
  return {
    id: String(f.id),
    toothLabel: f.tooth_label,
    finding: f.finding,
    confidence: f.confidence,
    accepted: f.accepted,
    polygon: f.polygon.map((p) => ({ x: p.x, y: p.y })),
    layer: f.layer,
  };
}

export function uiFindingToCreatePayload(f: Omit<Finding, "id">): FindingCreatePayload {
  return {
    tooth_label: f.toothLabel,
    finding: f.finding,
    confidence: f.confidence,
    accepted: f.accepted,
    polygon: f.polygon.map((p) => ({ x: p.x, y: p.y })),
    layer: f.layer,
    source: "manual",
  };
}

export function uiPatchToApi(partial: Partial<Finding>): FindingUpdatePayload {
  const out: FindingUpdatePayload = {};
  if (partial.toothLabel !== undefined) out.tooth_label = partial.toothLabel;
  if (partial.finding !== undefined) out.finding = partial.finding;
  if (partial.confidence !== undefined) out.confidence = partial.confidence;
  if (partial.accepted !== undefined) out.accepted = partial.accepted;
  if (partial.polygon !== undefined) out.polygon = partial.polygon.map((p) => ({ x: p.x, y: p.y }));
  if (partial.layer !== undefined) out.layer = partial.layer;
  return out;
}
