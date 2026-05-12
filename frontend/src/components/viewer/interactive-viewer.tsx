"use client";

import {
  appendAudit,
  completeReview,
  createFinding,
  deleteFinding,
  downloadAnalysisAuditExport,
  downloadAnalysisReportPdf,
  generateReport,
  fetchAnalysisImageBlob,
  getFindings,
  listAudit,
  patchFinding,
  previewReport,
} from "@/lib/api-client";
import type {
  AnalysisDetail,
  AuditEntryRead,
  ImageAssetKind,
  ReportGenerateRequest,
  ReportPreviewResponse,
} from "@/lib/api-types";
import { apiFindingToUi, uiFindingToCreatePayload, uiPatchToApi } from "@/lib/finding-mapper";
import type { Finding } from "@/lib/mock-data";
import { AUTH_CHANGED_EVENT } from "@/lib/auth";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CanvasStage } from "./canvas-stage";
import { FindingsPanel } from "./findings-panel";
import { ImageControls } from "./image-controls";
import { LayerToggles } from "./layer-toggles";
import { OverlayTabs } from "./overlay-tabs";
import { PropsPanel } from "./props-panel";
import { ShortcutsDialog } from "./shortcuts-dialog";
import { StatusBar } from "./status-bar";
import { ToolbarZoom } from "./toolbar-zoom";
import type { ViewerTool } from "./types";

const REVIEWER_STORAGE_KEY = "toothfairy.reviewer.v1";
const FINDING_OPTIONS = [
  "Caries",
  "Impacted",
  "Periapical Lesion",
  "Quadrant Region",
  "Manual Finding",
  "AI Tooth Mask",
];
const UNDO_CAP = 40;

type PolygonEditOp = {
  type: "polygon";
  findingId: string;
  before: { x: number; y: number }[];
  after: { x: number; y: number }[];
};

type HistoryOp = PolygonEditOp;

export function InteractiveViewer({
  analysisId,
  initialDetail,
}: {
  analysisId: string;
  initialDetail?: AnalysisDetail;
}) {
  const [threshold, setThreshold] = useState(0.25);
  const [layers, setLayers] = useState({
    quadrants: true,
    teeth: true,
    periapical: true,
    caries: true,
    impacted: true,
  });
  const [findings, setFindings] = useState<Finding[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [tool, setTool] = useState<ViewerTool>("select");
  const [draftPolygon, setDraftPolygon] = useState<{ x: number; y: number }[]>([]);
  const [boxDraft, setBoxDraft] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [reviewer, setReviewer] = useState("Dr. Demo");
  const [canvasImage, setCanvasImage] = useState<HTMLImageElement | null>(null);
  const [stageScale, setStageScale] = useState(1);
  const [stagePosition, setStagePosition] = useState({ x: 0, y: 0 });
  const [auditRows, setAuditRows] = useState<AuditEntryRead[]>([]);
  const [detail, setDetail] = useState<AnalysisDetail | null>(initialDetail ?? null);
  const [reviewError, setReviewError] = useState("");
  const [overlayKind, setOverlayKind] = useState<ImageAssetKind | "original">("original");
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
  const [gamma, setGamma] = useState(1);
  const [invert, setInvert] = useState(false);
  const [precisionMode, setPrecisionMode] = useState(false);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [pointerPx, setPointerPx] = useState<{ x: number; y: number } | null>(null);
  const [undoStack, setUndoStack] = useState<Finding[][]>([]);
  const [redoStack, setRedoStack] = useState<Finding[][]>([]);
  const [historyUndo, setHistoryUndo] = useState<HistoryOp[]>([]);
  const [historyRedo, setHistoryRedo] = useState<HistoryOp[]>([]);
  const [actionStateById, setActionStateById] = useState<Record<string, "idle" | "saving" | "error">>({});
  const [reportOpen, setReportOpen] = useState(false);
  const [reportBusy, setReportBusy] = useState(false);
  const [reportError, setReportError] = useState("");
  const [reportPreviewData, setReportPreviewData] = useState<ReportPreviewResponse | null>(null);
  const [reportDownloadEnabled, setReportDownloadEnabled] = useState(false);
  const [reportDraft, setReportDraft] = useState({
    clinical_summary: "",
    impression: "",
    recommendations: "",
    reviewer_confirmation: "",
  });
  const polygonEditStartRef = useRef<Record<string, { x: number; y: number }[]>>({});
  const auditRefreshTimer = useRef<number | null>(null);
  const findingsRef = useRef(findings);
  findingsRef.current = findings;

  type KeyboardHandlers = {
    beginPolygonEdit: (findingId: string) => void;
    undo: () => void;
    redo: () => void;
    zoomBy: (factor: number) => void;
    fitView: () => void;
    actualSize: () => void;
    fitSelected: () => void;
    selectedFinding: Finding | null;
    selectedId: string | null;
    onPolygonChange: (findingId: string, polygon: { x: number; y: number }[]) => void;
    persistPolygon: (findingId: string) => Promise<void>;
    updateFinding: (id: string, patch: Partial<Finding>) => Promise<void>;
    onEraseFinding: (id: string) => Promise<void>;
  };

  const kbRef = useRef<KeyboardHandlers>({
    beginPolygonEdit: () => {},
    undo: () => {},
    redo: () => {},
    zoomBy: () => {},
    fitView: () => {},
    actualSize: () => {},
    fitSelected: () => {},
    selectedFinding: null,
    selectedId: null,
    onPolygonChange: () => {},
    persistPolygon: async () => {},
    updateFinding: async () => {},
    onEraseFinding: async () => {},
  });

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(REVIEWER_STORAGE_KEY);
      if (raw?.trim()) setReviewer(raw.trim());
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(REVIEWER_STORAGE_KEY, reviewer);
    } catch {
      /* ignore */
    }
  }, [reviewer]);

  const [imageEpoch, setImageEpoch] = useState(0);
  useEffect(() => {
    const onAuth = () => setImageEpoch((n) => n + 1);
    window.addEventListener(AUTH_CHANGED_EVENT, onAuth);
    return () => window.removeEventListener(AUTH_CHANGED_EVENT, onAuth);
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    const blobHandle = { url: null as string | null };
    let alive = true;

    void (async () => {
      try {
        const blob = await fetchAnalysisImageBlob(analysisId, overlayKind, ac.signal);
        if (!alive) return;
        const url = URL.createObjectURL(blob);
        blobHandle.url = url;
        if (!alive) {
          URL.revokeObjectURL(url);
          blobHandle.url = null;
          return;
        }
        const img = new window.Image();
        img.onload = () => {
          if (!alive) return;
          setCanvasImage(img);
        };
        img.onerror = () => {
          if (!alive) return;
          toast.error("Could not load image");
        };
        img.src = url;
      } catch (e) {
        if (!alive || (e instanceof DOMException && e.name === "AbortError")) return;
        toast.error(e instanceof Error ? e.message : "Could not load image");
      }
    })();

    return () => {
      alive = false;
      ac.abort();
      if (blobHandle.url) {
        URL.revokeObjectURL(blobHandle.url);
        blobHandle.url = null;
      }
    };
  }, [analysisId, overlayKind, imageEpoch]);

  const loadFindings = useCallback(async () => {
    const rows = await getFindings(analysisId);
    setFindings(rows.map(apiFindingToUi));
  }, [analysisId]);

  const loadAudit = useCallback(async () => {
    const rows = await listAudit(analysisId);
    setAuditRows(rows);
  }, [analysisId]);

  const queueAuditRefresh = useCallback(
    (delayMs = 180) => {
      if (auditRefreshTimer.current !== null) {
        window.clearTimeout(auditRefreshTimer.current);
      }
      auditRefreshTimer.current = window.setTimeout(() => {
        auditRefreshTimer.current = null;
        void loadAudit();
      }, delayMs);
    },
    [loadAudit],
  );

  useEffect(() => {
    void loadFindings();
    void loadAudit();
  }, [loadAudit, loadFindings]);

  useEffect(
    () => () => {
      if (auditRefreshTimer.current !== null) {
        window.clearTimeout(auditRefreshTimer.current);
      }
    },
    [],
  );

  const pushUndo = useCallback(() => {
    setUndoStack((u) => [...u.slice(-UNDO_CAP + 1), findings]);
    setRedoStack([]);
  }, [findings]);

  const applyPolygonOp = useCallback((op: PolygonEditOp, direction: "undo" | "redo") => {
    const polygon = direction === "undo" ? op.before : op.after;
    setFindings((prev) => prev.map((f) => (f.id === op.findingId ? { ...f, polygon } : f)));
  }, []);

  const beginPolygonEdit = useCallback((findingId: string) => {
    const target = findingsRef.current.find((f) => f.id === findingId);
    if (!target) return;
    polygonEditStartRef.current[findingId] = target.polygon.map((p) => ({ ...p }));
  }, []);

  const undo = useCallback(() => {
    setHistoryUndo((ops) => {
      if (ops.length === 0) return ops;
      const op = ops[ops.length - 1];
      applyPolygonOp(op, "undo");
      setHistoryRedo((redo) => [...redo.slice(-UNDO_CAP + 1), op]);
      return ops.slice(0, -1);
    });
    setUndoStack((u) => {
      if (u.length === 0) return u;
      const prev = u[u.length - 1];
      setRedoStack((r) => [...r, findings]);
      setFindings(prev);
      return u.slice(0, -1);
    });
  }, [findings]);

  const redo = useCallback(() => {
    setHistoryRedo((ops) => {
      if (ops.length === 0) return ops;
      const op = ops[ops.length - 1];
      applyPolygonOp(op, "redo");
      setHistoryUndo((undoOps) => [...undoOps.slice(-UNDO_CAP + 1), op]);
      return ops.slice(0, -1);
    });
    setRedoStack((r) => {
      if (r.length === 0) return r;
      const next = r[r.length - 1];
      setUndoStack((u) => [...u, findings]);
      setFindings(next);
      return r.slice(0, -1);
    });
  }, [applyPolygonOp, findings]);

  const visibleFindings = useMemo(
    () =>
      findings.filter(
        (f) => {
          if (f.confidence < threshold) return false;
          if (f.layer === "quadrants") return layers.quadrants;
          if (f.layer === "periapical") return layers.periapical;
          if (f.layer === "teeth") {
            if (f.finding === "Caries") return layers.caries;
            if (f.finding === "Impacted") return layers.impacted;
            return layers.teeth;
          }
          return true;
        },
      ),
    [findings, layers, threshold],
  );

  const selectedFinding = findings.find((f) => f.id === selectedId) ?? null;
  const imgW = canvasImage?.width ?? 1;
  const imgH = canvasImage?.height ?? 1;

  const fitView = useCallback(() => {
    const host = document.getElementById("viewer-canvas-host");
    const cw = host?.clientWidth ?? 800;
    const ch = host?.clientHeight ?? 560;
    const fit = Math.min(cw / imgW, ch / imgH);
    const s = Number.isFinite(fit) && fit > 0 ? Math.min(8, Math.max(0.1, fit)) : 1;
    setStageScale(s);
    setStagePosition({ x: (cw - imgW * s) / 2, y: (ch - imgH * s) / 2 });
  }, [imgH, imgW]);

  const actualSize = useCallback(() => {
    const host = document.getElementById("viewer-canvas-host");
    const cw = host?.clientWidth ?? 800;
    const ch = host?.clientHeight ?? 560;
    setStageScale(1);
    setStagePosition({ x: (cw - imgW) / 2, y: (ch - imgH) / 2 });
  }, [imgH, imgW]);

  const fitSelected = useCallback(() => {
    if (!selectedFinding?.polygon.length) return;
    const host = document.getElementById("viewer-canvas-host");
    const cw = host?.clientWidth ?? 800;
    const ch = host?.clientHeight ?? 560;
    const xs = selectedFinding.polygon.map((p) => p.x);
    const ys = selectedFinding.polygon.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const bw = maxX - minX;
    const bh = maxY - minY;
    const pad = 40;
    const s = Math.min(8, Math.max(0.1, Math.min(cw / (bw + pad * 2), ch / (bh + pad * 2))));
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    setStageScale(s);
    setStagePosition({ x: cw / 2 - cx * s, y: ch / 2 - cy * s });
  }, [selectedFinding]);

  const centerFinding = useCallback((id: string) => {
    const target = findings.find((f) => f.id === id);
    if (!target?.polygon.length) return;
    const host = document.getElementById("viewer-canvas-host");
    const cw = host?.clientWidth ?? 800;
    const ch = host?.clientHeight ?? 560;
    const xs = target.polygon.map((p) => p.x);
    const ys = target.polygon.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    setSelectedId(id);
    setStagePosition({ x: cw / 2 - cx * stageScale, y: ch / 2 - cy * stageScale });
  }, [findings, stageScale]);

  useEffect(() => {
    if (!canvasImage) return;
    fitView();
  }, [canvasImage, fitView]);

  const persistPolygon = async (findingId: string) => {
    const poly = findingsRef.current.find((f) => f.id === findingId)?.polygon;
    if (!poly) return;
    const before = polygonEditStartRef.current[findingId];
    if (before && JSON.stringify(before) !== JSON.stringify(poly)) {
      const op: PolygonEditOp = {
        type: "polygon",
        findingId,
        before,
        after: poly.map((p) => ({ ...p })),
      };
      setHistoryUndo((ops) => [...ops.slice(-UNDO_CAP + 1), op]);
      setHistoryRedo([]);
    }
    delete polygonEditStartRef.current[findingId];
    setSaveState("saving");
    try {
      setActionStateById((prev) => ({ ...prev, [findingId]: "saving" }));
      await patchFinding(findingId, uiPatchToApi({ polygon: poly }), reviewer.trim() || "Unknown Reviewer");
      setSaveState("saved");
      window.setTimeout(() => setSaveState("idle"), 1600);
      queueAuditRefresh();
      setActionStateById((prev) => ({ ...prev, [findingId]: "idle" }));
    } catch (e) {
      setSaveState("error");
      toast.error(e instanceof Error ? e.message : "Save failed");
      setActionStateById((prev) => ({ ...prev, [findingId]: "error" }));
      await loadFindings();
    }
  };

  const onPolygonChange = useCallback((findingId: string, polygon: { x: number; y: number }[]) => {
    setFindings((prev) => prev.map((f) => (f.id === findingId ? { ...f, polygon } : f)));
  }, []);

  const insertVertex = useCallback(
    async (findingId: string, afterIndex: number, point: { x: number; y: number }) => {
      const f = findings.find((x) => x.id === findingId);
      if (!f || f.polygon.length >= 200) return;
      const poly = [...f.polygon];
      poly.splice(afterIndex + 1, 0, point);
      pushUndo();
      setFindings((prev) => prev.map((x) => (x.id === findingId ? { ...x, polygon: poly } : x)));
      try {
        await patchFinding(findingId, uiPatchToApi({ polygon: poly }), reviewer.trim() || "Unknown Reviewer");
        queueAuditRefresh();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Failed");
        await loadFindings();
      }
    },
    [findings, loadFindings, pushUndo, queueAuditRefresh, reviewer],
  );

  const deleteVertex = useCallback(
    async (findingId: string, vertexIndex: number) => {
      const f = findings.find((x) => x.id === findingId);
      if (!f || f.polygon.length <= 3) return;
      const poly = f.polygon.filter((_, i) => i !== vertexIndex);
      pushUndo();
      setFindings((prev) => prev.map((x) => (x.id === findingId ? { ...x, polygon: poly } : x)));
      try {
        await patchFinding(findingId, uiPatchToApi({ polygon: poly }), reviewer.trim() || "Unknown Reviewer");
        queueAuditRefresh();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Failed");
        await loadFindings();
      }
    },
    [findings, loadFindings, pushUndo, queueAuditRefresh, reviewer],
  );

  const updateFinding = async (id: string, patch: Partial<Finding>) => {
    const prev = findings;
    setFindings((p) => p.map((f) => (f.id === id ? { ...f, ...patch } : f)));
    try {
      setActionStateById((states) => ({ ...states, [id]: "saving" }));
      const updated = await patchFinding(id, uiPatchToApi(patch), reviewer.trim() || "Unknown Reviewer");
      setFindings((rows) => rows.map((f) => (f.id === id ? apiFindingToUi(updated) : f)));
      queueAuditRefresh();
      setActionStateById((states) => ({ ...states, [id]: "idle" }));
    } catch (e) {
      setFindings(prev);
      setActionStateById((states) => ({ ...states, [id]: "error" }));
      toast.error(e instanceof Error ? e.message : "Update failed");
    }
  };

  const onEraseFinding = async (id: string) => {
    pushUndo();
    const prev = findings;
    setFindings((p) => p.filter((f) => f.id !== id));
    setSelectedId(null);
    try {
      setActionStateById((states) => ({ ...states, [id]: "saving" }));
      await deleteFinding(id, reviewer.trim() || "Unknown Reviewer");
      queueAuditRefresh();
      setActionStateById((states) => ({ ...states, [id]: "idle" }));
    } catch (e) {
      setFindings(prev);
      setActionStateById((states) => ({ ...states, [id]: "error" }));
      toast.error(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const commitDraft = async () => {
    if (draftPolygon.length < 3) return;
    const base: Omit<Finding, "id"> = {
      toothLabel: "FDI-XX",
      finding: "Manual Finding",
      confidence: 1,
      accepted: true,
      polygon: draftPolygon,
      layer: "teeth",
    };
    try {
      const created = await createFinding(analysisId, uiFindingToCreatePayload(base), reviewer.trim() || "Unknown Reviewer");
      pushUndo();
      setFindings((p) => [...p, apiFindingToUi(created)]);
      setDraftPolygon([]);
      setTool("select");
      queueAuditRefresh();
      toast.success("Finding created");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Create failed");
    }
  };

  const onBoxComplete = async (rect: { x: number; y: number; w: number; h: number }) => {
    const { x, y, w, h } = rect;
    const poly = [
      { x, y },
      { x: x + w, y },
      { x: x + w, y: y + h },
      { x, y: y + h },
    ];
    try {
      const created = await createFinding(
        analysisId,
        uiFindingToCreatePayload({
          toothLabel: "FDI-XX",
          finding: "Manual Finding",
          confidence: 1,
          accepted: true,
          polygon: poly,
          layer: "periapical",
        }),
        reviewer.trim() || "Unknown Reviewer",
      );
      pushUndo();
      setFindings((p) => [...p, apiFindingToUi(created)]);
      setTool("select");
      queueAuditRefresh();
      toast.success("Box finding created");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Create failed");
    }
  };

  const markReviewComplete = async () => {
    try {
      const updated = await completeReview(analysisId, reviewer.trim() || "Unknown Reviewer");
      setDetail(updated);
      setReviewError("");
      queueAuditRefresh();
      toast.success("Review complete");
    } catch (e) {
      setReviewError(e instanceof Error ? e.message : "Could not complete review.");
    }
  };

  const onReportFieldChange = useCallback(
    (
      field: "clinical_summary" | "impression" | "recommendations" | "reviewer_confirmation",
      value: string,
    ) => {
      setReportDraft((prev) => ({ ...prev, [field]: value }));
    },
    [],
  );

  const doPreviewReport = async () => {
    setReportBusy(true);
    setReportError("");
    try {
      const payload = {
        clinical_summary: reportDraft.clinical_summary,
        impression: reportDraft.impression,
        recommendations: reportDraft.recommendations,
        reviewer_confirmation: reportDraft.reviewer_confirmation || null,
        include_images: true,
      };
      const preview = await previewReport(analysisId, payload);
      setReportPreviewData(preview);
    } catch (e) {
      setReportError(e instanceof Error ? e.message : "Could not preview report");
    } finally {
      setReportBusy(false);
    }
  };

  const doGenerateReport = async () => {
    if (detail?.status !== "Reviewed" && detail?.status !== "Report Generated") return;
    setReportBusy(true);
    setReportError("");
    try {
      const payload: ReportGenerateRequest = {
        clinical_summary: reportDraft.clinical_summary,
        impression: reportDraft.impression,
        recommendations: reportDraft.recommendations,
        reviewer_confirmation: reportDraft.reviewer_confirmation || null,
        include_images: true,
        reviewer: reviewer.trim() || "Unknown Reviewer",
      };
      const generated = await generateReport(analysisId, payload);
      setDetail((prev) => (prev ? { ...prev, status: generated.status } : prev));
      setReportDownloadEnabled(true);
      await appendAudit(analysisId, {
        reviewer: reviewer.trim() || "Unknown Reviewer",
        action: `Generated report ${generated.filename}`,
        action_type: "export",
      });
      queueAuditRefresh();
      toast.success("PDF generated");
    } catch (e) {
      setReportError(e instanceof Error ? e.message : "Could not generate report");
    } finally {
      setReportBusy(false);
    }
  };

  const doDownloadReport = useCallback(async () => {
    try {
      await downloadAnalysisReportPdf(analysisId);
      toast.success("Download started");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not download PDF");
    }
  }, [analysisId]);

  const zoomBy = useCallback((factor: number) => {
    setStageScale((s) => {
      const next = Math.min(8, Math.max(0.1, s * factor));
      setStagePosition((pos) => {
        const host = document.getElementById("viewer-canvas-host");
        const cw = host?.clientWidth ?? 800;
        const ch = host?.clientHeight ?? 560;
        const cx = cw / 2;
        const cy = ch / 2;
        const wx = (cx - pos.x) / s;
        const wy = (cy - pos.y) / s;
        return { x: cx - wx * next, y: cy - wy * next };
      });
      return next;
    });
  }, []);

  kbRef.current = {
    beginPolygonEdit,
    undo,
    redo,
    zoomBy,
    fitView,
    actualSize,
    fitSelected,
    selectedFinding,
    selectedId,
    onPolygonChange,
    persistPolygon,
    updateFinding,
    onEraseFinding,
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.closest("input") || el.closest("textarea") || el.closest('[role="combobox"]'))) return;
      const k = kbRef.current;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) k.redo();
        else k.undo();
        return;
      }
      if (e.key === "Escape") {
        setDraftPolygon([]);
        setSelectedId(null);
      }
      if (e.key.toLowerCase() === "v") setTool("select");
      if (e.key.toLowerCase() === "h") setTool("pan");
      if (e.key.toLowerCase() === "p") setTool("drawPolygon");
      if (e.key.toLowerCase() === "b") setTool("drawBox");
      if (e.key.toLowerCase() === "e") setTool("erase");
      if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        k.zoomBy(1.12);
      }
      if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        k.zoomBy(1 / 1.12);
      }
      if (e.key === "0") {
        e.preventDefault();
        k.fitView();
      }
      if (e.key === "1") {
        e.preventDefault();
        k.actualSize();
      }
      if (e.key.toLowerCase() === "f") {
        e.preventDefault();
        k.fitSelected();
      }
      const sf = k.selectedFinding;
      if (sf && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
        e.preventDefault();
        k.beginPolygonEdit(sf.id);
        const step = e.shiftKey ? 10 : 1;
        const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
        const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
        k.onPolygonChange(sf.id, sf.polygon.map((p) => ({ x: p.x + dx, y: p.y + dy })));
        void k.persistPolygon(sf.id);
      }
      const sid = k.selectedId;
      if (sid && e.key.toLowerCase() === "a") void k.updateFinding(sid, { accepted: true });
      if (sid && e.key.toLowerCase() === "r") void k.updateFinding(sid, { accepted: false });
      if (sid && e.key === "Delete") void k.onEraseFinding(sid);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Shift") setPrecisionMode(true);
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") setPrecisionMode(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onUp);
    };
  }, []);

  const zoomPercent = Math.round(stageScale * 100);

  return (
    <section className="flex h-[calc(100vh-11.5rem)] min-h-[680px] flex-col gap-4 overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Review scan</h1>
          <p className="text-sm text-muted-foreground">
            Analysis <span className="font-mono text-xs">{analysisId}</span> · Status{" "}
            <span className="font-medium text-foreground">{detail?.status ?? "—"}</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ShortcutsDialog />
        </div>
      </div>

      <div className="grid min-h-0 h-full flex-1 gap-4 xl:grid-cols-[minmax(260px,320px)_minmax(0,1fr)_minmax(260px,340px)]">
        <aside className="flex min-h-0 flex-col gap-3">
          <FindingsPanel
            visibleRows={visibleFindings}
            selectedId={selectedId}
            onSelect={(id) => setSelectedId(id)}
            onAccept={(id) => void updateFinding(id, { accepted: true })}
            onReject={(id) => void updateFinding(id, { accepted: false })}
            onHover={setHoveredId}
            onCenter={centerFinding}
            onReclassify={(id, label) => void updateFinding(id, { finding: label })}
            onDelete={(id) => void onEraseFinding(id)}
            reclassifyOptions={FINDING_OPTIONS}
            actionStateById={actionStateById}
          />
        </aside>

        <div className="flex min-h-0 min-w-0 flex-col gap-3">
          <OverlayTabs value={overlayKind} onChange={setOverlayKind} />
          <ToolbarZoom
            zoomPercent={zoomPercent}
            onZoomIn={() => zoomBy(1.12)}
            onZoomOut={() => zoomBy(1 / 1.12)}
            onFit={fitView}
            onActual={actualSize}
            onFitSelected={fitSelected}
            disabledFitSelected={!selectedFinding?.polygon.length}
          />
          <div id="viewer-canvas-host" className="relative flex min-h-0 h-full flex-1 flex-col overflow-hidden rounded-xl border border-border bg-viewer-canvas">
            <CanvasStage
              findings={findings}
              visibleFindings={visibleFindings}
              selectedId={selectedId}
              onSelectId={(id) => setSelectedId(id)}
              canvasImage={canvasImage}
              stageScale={stageScale}
              setStageScale={setStageScale}
              stagePosition={stagePosition}
              setStagePosition={setStagePosition}
              tool={tool}
              draftPolygon={draftPolygon}
              onAddDraftPoint={(p) => setDraftPolygon((d) => [...d, p])}
              onCloseDraft={() => void commitDraft()}
              boxDraft={boxDraft}
              setBoxDraft={setBoxDraft}
              onBoxComplete={(r) => void onBoxComplete(r)}
              brightness={brightness}
              contrast={contrast}
              gamma={gamma}
              invert={invert}
              precisionMode={precisionMode}
              onPolygonEditStart={beginPolygonEdit}
              onPolygonChange={onPolygonChange}
              onPolygonCommit={(id) => void persistPolygon(id)}
              onInsertVertex={(id, idx, pt) => void insertVertex(id, idx, pt)}
              onDeleteVertex={(id, vi) => void deleteVertex(id, vi)}
              onEraseFinding={(id) => void onEraseFinding(id)}
              onScenePointer={setPointerPx}
              hoveredId={hoveredId}
            />
          </div>
          <StatusBar
            pointerPx={pointerPx}
            zoomPercent={zoomPercent}
            saveState={saveState}
            reviewer={reviewer}
            onReviewerChange={setReviewer}
            selectedSummary={
              selectedFinding
                ? `${selectedFinding.toothLabel} · ${selectedFinding.finding} · ${Math.round(selectedFinding.confidence * 100)}%`
                : null
            }
          />
        </div>

        <aside className="flex min-h-0 flex-col gap-4 overflow-y-auto pb-8">
          <LayerToggles layers={layers} onChange={setLayers} />
          <ImageControls
            brightness={brightness}
            contrast={contrast}
            gamma={gamma}
            invert={invert}
            onBrightness={setBrightness}
            onContrast={setContrast}
            onGamma={setGamma}
            onInvert={setInvert}
            onReset={() => {
              setBrightness(0);
              setContrast(0);
              setGamma(1);
              setInvert(false);
            }}
          />
          <PropsPanel
            tool={tool}
            onTool={setTool}
            threshold={threshold}
            onThreshold={setThreshold}
            findingOptions={FINDING_OPTIONS}
            selectedFinding={selectedFinding}
            onReclassify={(lab) => selectedId && void updateFinding(selectedId, { finding: lab })}
            onChangeToothLabel={(lab) => selectedId && void updateFinding(selectedId, { toothLabel: lab })}
            onDelete={() => selectedId && void onEraseFinding(selectedId)}
            onUndo={undo}
            onRedo={redo}
            undoDisabled={undoStack.length === 0 && historyUndo.length === 0}
            redoDisabled={redoStack.length === 0 && historyRedo.length === 0}
            onMarkReviewComplete={() => void markReviewComplete()}
            onExportAudit={() => {
              void downloadAnalysisAuditExport(analysisId).catch((e) =>
                toast.error(e instanceof Error ? e.message : "Could not export audit"),
              );
            }}
            onReportFieldChange={onReportFieldChange}
            onPreviewReport={() => void doPreviewReport()}
            onGenerateReport={() => void doGenerateReport()}
            onDownloadReport={doDownloadReport}
            reviewError={reviewError}
            pdfDisabled={detail?.status !== "Reviewed" && detail?.status !== "Report Generated"}
            reportOpen={reportOpen}
            onReportOpenChange={setReportOpen}
            reportDraft={reportDraft}
            reportPreview={reportPreviewData}
            reportBusy={reportBusy}
            reportError={reportError}
            reportDownloadEnabled={reportDownloadEnabled}
            precisionMode={precisionMode}
            onPrecisionMode={setPrecisionMode}
          />
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Audit entries: {auditRows.length}. AI assists — you confirm every clinical decision.
          </p>
        </aside>
      </div>
    </section>
  );
}
