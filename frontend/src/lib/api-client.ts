import { getStoredAccessToken } from "@/lib/auth";

import type {
  AnalysisCreateResponse,
  AnalysisDetail,
  AnalysisListItem,
  AuditEntryCreatePayload,
  AuditEntryRead,
  FindingCreatePayload,
  FindingRead,
  FindingUpdatePayload,
  ImageAssetKind,
  ReportDraftPayload,
  ReportGenerateRequest,
  ReportGenerateResponse,
  ReportPreviewResponse,
} from "./api-types";

const DEFAULT_BASE = "http://localhost:8000";

function baseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? DEFAULT_BASE;
}

function apiV1(path: string): string {
  return `${baseUrl()}/api/v1${path.startsWith("/") ? path : `/${path}`}`;
}

/** Optional Bearer token (set via sign-in; see ``@/lib/auth``). */
function authHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = getStoredAccessToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

const fetchDefaults: RequestInit = { credentials: "include" };

async function parseError(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return JSON.stringify(j.detail);
    return res.statusText;
  } catch {
    return res.statusText;
  }
}

export async function createAnalysis(form: FormData): Promise<AnalysisCreateResponse> {
  const res = await fetch(apiV1("/analyses"), {
    method: "POST",
    body: form,
    ...fetchDefaults,
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<AnalysisCreateResponse>;
}

export async function getAnalysis(id: string): Promise<AnalysisDetail> {
  const res = await fetch(apiV1(`/analyses/${id}`), {
    cache: "no-store",
    ...fetchDefaults,
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<AnalysisDetail>;
}

export async function listAnalyses(limit = 100): Promise<AnalysisListItem[]> {
  const res = await fetch(apiV1(`/analyses?limit=${limit}`), {
    cache: "no-store",
    ...fetchDefaults,
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<AnalysisListItem[]>;
}

export async function getFindings(analysisId: string): Promise<FindingRead[]> {
  const res = await fetch(apiV1(`/analyses/${analysisId}/findings`), {
    cache: "no-store",
    ...fetchDefaults,
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<FindingRead[]>;
}

export function imageUrl(analysisId: string, kind: ImageAssetKind | "original"): string {
  return `${apiV1(`/analyses/${analysisId}/image`)}?kind=${encodeURIComponent(kind)}`;
}

/**
 * Load an analysis image with the same auth as other API calls (Bearer + guest cookie).
 * Use this instead of assigning {@link imageUrl} to ``<img src>`` — browsers do not send
 * custom headers on plain image requests, so JWT-only sessions would always 404.
 */
export async function fetchAnalysisImageBlob(
  analysisId: string,
  kind: ImageAssetKind | "original",
  signal?: AbortSignal,
): Promise<Blob> {
  const res = await fetch(imageUrl(analysisId, kind), {
    ...fetchDefaults,
    cache: "no-store",
    headers: { ...authHeaders() },
    signal,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.blob();
}

export async function createFinding(
  analysisId: string,
  payload: FindingCreatePayload,
  reviewer: string,
): Promise<FindingRead> {
  const res = await fetch(apiV1(`/analyses/${analysisId}/findings`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Reviewer": reviewer,
      ...authHeaders(),
    },
    body: JSON.stringify(payload),
    ...fetchDefaults,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<FindingRead>;
}

export async function patchFinding(
  findingId: string,
  payload: FindingUpdatePayload,
  reviewer: string,
): Promise<FindingRead> {
  const res = await fetch(apiV1(`/findings/${findingId}`), {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      "X-Reviewer": reviewer,
      ...authHeaders(),
    },
    body: JSON.stringify(payload),
    ...fetchDefaults,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<FindingRead>;
}

export async function deleteFinding(findingId: string, reviewer: string): Promise<void> {
  const res = await fetch(apiV1(`/findings/${findingId}`), {
    method: "DELETE",
    headers: { "X-Reviewer": reviewer, ...authHeaders() },
    ...fetchDefaults,
  });
  if (!res.ok) throw new Error(await parseError(res));
}

export async function listAudit(analysisId: string): Promise<AuditEntryRead[]> {
  const res = await fetch(apiV1(`/analyses/${analysisId}/audit`), {
    cache: "no-store",
    ...fetchDefaults,
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(await parseError(res));
  const raw = (await res.json()) as Array<
    Record<string, unknown> & { target_finding_id?: string | null }
  >;
  return raw.map((row) => ({
    id: String(row.id),
    reviewer: String(row.reviewer),
    action: String(row.action),
    action_type: row.action_type as AuditEntryRead["action_type"],
    target_id: (row.target_id as string | null | undefined) ?? (row.target_finding_id as string | null) ?? null,
    before: (row.before as Record<string, unknown> | null) ?? null,
    after: (row.after as Record<string, unknown> | null) ?? null,
    timestamp: String(row.timestamp),
  }));
}

export async function appendAudit(analysisId: string, payload: AuditEntryCreatePayload): Promise<AuditEntryRead> {
  const res = await fetch(apiV1(`/analyses/${analysisId}/audit`), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
    ...fetchDefaults,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<AuditEntryRead>;
}

export async function completeReview(analysisId: string, reviewer: string): Promise<AnalysisDetail> {
  const res = await fetch(apiV1(`/analyses/${analysisId}/complete-review`), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ reviewer }),
    ...fetchDefaults,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<AnalysisDetail>;
}

export function auditExportUrl(analysisId: string): string {
  return apiV1(`/analyses/${analysisId}/audit/export`);
}

export async function previewReport(
  analysisId: string,
  payload: ReportDraftPayload,
): Promise<ReportPreviewResponse> {
  const res = await fetch(apiV1(`/analyses/${analysisId}/report/preview`), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
    ...fetchDefaults,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<ReportPreviewResponse>;
}

export async function generateReport(
  analysisId: string,
  payload: ReportGenerateRequest,
): Promise<ReportGenerateResponse> {
  const res = await fetch(apiV1(`/analyses/${analysisId}/report/generate`), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
    ...fetchDefaults,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<ReportGenerateResponse>;
}

export function reportDownloadUrl(analysisId: string): string {
  return apiV1(`/analyses/${analysisId}/report/download`);
}

/**
 * GET with Bearer + cookies, then trigger a file save. Use instead of ``window.open(reportDownloadUrl)`` —
 * plain navigations do not send ``Authorization``.
 */
export async function downloadAnalysisReportPdf(analysisId: string): Promise<void> {
  const url = reportDownloadUrl(analysisId);
  const res = await fetch(url, {
    ...fetchDefaults,
    cache: "no-store",
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(await parseError(res));
  let filename = `report-${analysisId}.pdf`;
  const cd = res.headers.get("Content-Disposition");
  if (cd) {
    const star = /filename\*=UTF-8''([^;\s]+)/i.exec(cd);
    const quoted = /filename="([^"]+)"/i.exec(cd);
    if (star) filename = decodeURIComponent(star[1]);
    else if (quoted) filename = quoted[1];
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
}

/** Same auth issue as report PDF: use fetch + blob instead of ``window.open(auditExportUrl)``. */
export async function downloadAnalysisAuditExport(analysisId: string): Promise<void> {
  const url = auditExportUrl(analysisId);
  const res = await fetch(url, {
    ...fetchDefaults,
    cache: "no-store",
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(await parseError(res));
  const filename = `audit-${analysisId}.json`;
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
}

/** Exchange user id for a JWT (requires ``TOOTHFAIRY_AUTH_DEV_LOGIN_ENABLED`` on the API). */
export async function requestDevToken(userId: string): Promise<{ access_token: string; token_type: string }> {
  const res = await fetch(apiV1("/auth/token"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<{ access_token: string; token_type: string }>;
}
