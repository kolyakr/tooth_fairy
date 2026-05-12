"use client";

import Konva from "konva";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Image as KonvaImage, Layer, Line, Stage } from "react-konva";

import type { Finding } from "@/lib/mock-data";

import { fillForFinding, strokeForFinding } from "./colors";
import type { ViewerTool } from "./types";

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 8;
const CLOSE_RADIUS = 14;
const GRID_ZOOM_THRESHOLD = 4;

export type CanvasStageProps = {
  findings: Finding[];
  visibleFindings: Finding[];
  selectedId: string | null;
  onSelectId: (id: string | null) => void;
  canvasImage: HTMLImageElement | null;
  stageScale: number;
  setStageScale: React.Dispatch<React.SetStateAction<number>>;
  stagePosition: { x: number; y: number };
  setStagePosition: React.Dispatch<React.SetStateAction<{ x: number; y: number }>>;
  tool: ViewerTool;
  draftPolygon: { x: number; y: number }[];
  onAddDraftPoint: (p: { x: number; y: number }) => void;
  onCloseDraft: () => void;
  boxDraft: { x: number; y: number; w: number; h: number } | null;
  setBoxDraft: React.Dispatch<
    React.SetStateAction<{ x: number; y: number; w: number; h: number } | null>
  >;
  onBoxComplete?: (rect: { x: number; y: number; w: number; h: number }) => void;
  brightness: number;
  contrast: number;
  /** Mid-tone emphasis (1 = neutral). Applied as extra contrast latitude on top of Contrast. */
  gamma: number;
  invert: boolean;
  precisionMode: boolean;
  onPolygonEditStart?: (findingId: string) => void;
  onPolygonChange: (findingId: string, polygon: { x: number; y: number }[]) => void;
  onPolygonCommit: (findingId: string) => void;
  onInsertVertex?: (findingId: string, afterIndex: number, point: { x: number; y: number }) => void;
  onDeleteVertex?: (findingId: string, vertexIndex: number) => void;
  onEraseFinding?: (findingId: string) => void;
  onScenePointer?: (point: { x: number; y: number } | null) => void;
  hoveredId?: string | null;
};

export function CanvasStage({
  findings,
  visibleFindings,
  selectedId,
  onSelectId,
  canvasImage,
  stageScale,
  setStageScale,
  stagePosition,
  setStagePosition,
  tool,
  draftPolygon,
  onAddDraftPoint,
  onCloseDraft,
  boxDraft,
  setBoxDraft,
  onBoxComplete,
  brightness,
  contrast,
  gamma,
  invert,
  precisionMode,
  onPolygonEditStart,
  onPolygonChange,
  onPolygonCommit,
  onInsertVertex,
  onDeleteVertex,
  onEraseFinding,
  onScenePointer,
  hoveredId,
}: CanvasStageProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const imageRef = useRef<Konva.Image>(null);
  const [size, setSize] = useState({ w: 800, h: 560 });
  const spaceDown = useRef(false);
  const panning = useRef(false);
  const panStart = useRef<{ x: number; y: number } | null>(null);
  const posStart = useRef<{ x: number; y: number } | null>(null);
  const dragPoly = useRef<{ id: string; last: { x: number; y: number } | null } | null>(null);
  const edgeLast = useRef<{ x: number; y: number } | null>(null);
  const boxDrag = useRef<{ sx: number; sy: number } | null>(null);
  const pointerFrame = useRef<number | null>(null);
  const pendingPointer = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(320, Math.floor(r.width)), h: Math.max(280, Math.floor(r.height)) });
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setSize({ w: Math.max(320, Math.floor(r.width)), h: Math.max(280, Math.floor(r.height)) });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const kd = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceDown.current = true;
    };
    const ku = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceDown.current = false;
    };
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);
    return () => {
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
    };
  }, []);

  const imgW = canvasImage?.width ?? 1;
  const imgH = canvasImage?.height ?? 1;

  const clampCamera = useCallback(
    (position: { x: number; y: number }, scale: number) => {
      const margin = 0.08;
      const mx = size.w * margin;
      const my = size.h * margin;
      const sw = imgW * scale;
      const sh = imgH * scale;
      const centeredX = (size.w - sw) / 2;
      const centeredY = (size.h - sh) / 2;
      if (sw <= size.w - mx * 2 && sh <= size.h - my * 2) {
        return { x: centeredX, y: centeredY };
      }
      const minX = sw <= size.w - mx * 2 ? centeredX : size.w - sw - mx;
      const maxX = sw <= size.w - mx * 2 ? centeredX : mx;
      const minY = sh <= size.h - my * 2 ? centeredY : size.h - sh - my;
      const maxY = sh <= size.h - my * 2 ? centeredY : my;
      return {
        x: Math.max(minX, Math.min(maxX, position.x)),
        y: Math.max(minY, Math.min(maxY, position.y)),
      };
    },
    [imgH, imgW, size.h, size.w],
  );

  const toScene = useCallback(
    (pointer: { x: number; y: number }) => ({
      x: (pointer.x - stagePosition.x) / stageScale,
      y: (pointer.y - stagePosition.y) / stageScale,
    }),
    [stagePosition.x, stagePosition.y, stageScale],
  );

  useEffect(() => {
    const img = imageRef.current;
    if (!img || !canvasImage) return;
    const Fl = Konva.Filters as typeof Konva.Filters & { Invert?: (typeof Konva.Filters)["Brighten"] };
    const chain = [Konva.Filters.Brighten, Konva.Filters.Contrast];
    if (invert && Fl.Invert) chain.push(Fl.Invert);
    img.clearCache();
    img.filters(chain);
    img.brightness(brightness);
    img.contrast(contrast + (gamma - 1) * 55);
    img.cache();
    img.getLayer()?.batchDraw();
  }, [brightness, canvasImage, contrast, gamma, invert]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    stage.container().querySelectorAll("canvas").forEach((canvas) => {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.imageSmoothingEnabled = stageScale < 2;
    });
  }, [stageScale]);

  useEffect(() => {
    return () => {
      if (pointerFrame.current != null) {
        window.cancelAnimationFrame(pointerFrame.current);
        pointerFrame.current = null;
      }
    };
  }, []);

  const selectedFinding = useMemo(
    () => findings.find((f) => f.id === selectedId) ?? null,
    [findings, selectedId],
  );

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const scaleBy = 1.08;
    const oldScale = stageScale;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const isPinch = e.evt.ctrlKey;
    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const factor = isPinch ? 1.03 : scaleBy;
    const nextScale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, oldScale * (direction > 0 ? factor : 1 / factor)));
    const mousePointTo = {
      x: (pointer.x - stagePosition.x) / oldScale,
      y: (pointer.y - stagePosition.y) / oldScale,
    };
    const newPos = {
      x: pointer.x - mousePointTo.x * nextScale,
      y: pointer.y - mousePointTo.y * nextScale,
    };
    setStageScale(nextScale);
    setStagePosition(clampCamera(newPos, nextScale));
  };

  const canPan = tool === "pan" || spaceDown.current;

  const onStageMouseDown = (ev: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const scene = toScene(pointer);
    const clickedEmptySpace =
      ev.target === stage || ev.target.getClassName() === "Layer" || ev.target.getClassName() === "Image";

    if (ev.evt.button === 1 || canPan) {
      panning.current = true;
      panStart.current = pointer;
      posStart.current = { ...stagePosition };
      return;
    }

    if (tool === "select" && clickedEmptySpace) {
      onSelectId(null);
      return;
    }

    if (tool === "drawBox") {
      boxDrag.current = { sx: scene.x, sy: scene.y };
      setBoxDraft({ x: scene.x, y: scene.y, w: 0, h: 0 });
      return;
    }

    if (tool === "drawPolygon") {
      if (draftPolygon.length >= 3) {
        const first = draftPolygon[0];
        if (Math.hypot(scene.x - first.x, scene.y - first.y) <= CLOSE_RADIUS) {
          onCloseDraft();
          return;
        }
      }
      onAddDraftPoint(scene);
    }

    if (tool === "erase") {
      /* handled by line hit */
    }
  };

  const onStageMouseMove = (ev?: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    pendingPointer.current = toScene(pointer);
    if (onScenePointer && pointerFrame.current == null) {
      pointerFrame.current = window.requestAnimationFrame(() => {
        pointerFrame.current = null;
        onScenePointer(pendingPointer.current);
      });
    }

    if (panning.current && panStart.current && posStart.current) {
      const dx = pointer.x - panStart.current.x;
      const dy = pointer.y - panStart.current.y;
      setStagePosition(clampCamera({ x: posStart.current.x + dx, y: posStart.current.y + dy }, stageScale));
      return;
    }

    if (tool === "drawBox" && boxDrag.current && boxDraft) {
      const scene = toScene(pointer);
      const x0 = boxDrag.current.sx;
      const y0 = boxDrag.current.sy;
      const x = Math.min(x0, scene.x);
      const y = Math.min(y0, scene.y);
      const w = Math.abs(scene.x - x0);
      const h = Math.abs(scene.y - y0);
      setBoxDraft({ x, y, w, h });
    }
  };

  const onStageMouseUp = () => {
    if (panning.current) {
      panning.current = false;
      panStart.current = null;
      posStart.current = null;
    }
    if (tool === "drawBox" && boxDraft && boxDraft.w > 4 && boxDraft.h > 4) {
      onBoxComplete?.(boxDraft);
    }
    if (tool === "drawBox") {
      boxDrag.current = null;
      setBoxDraft(null);
    }
  };

  const gridLines = useMemo(() => {
    if (stageScale < GRID_ZOOM_THRESHOLD || !canvasImage) return null;
    const step = 50;
    const lines: number[] = [];
    for (let x = 0; x <= imgW; x += step) lines.push(x);
    for (let y = 0; y <= imgH; y += step) lines.push(-y);
    return { step, imgW, imgH };
  }, [canvasImage, imgH, imgW, stageScale]);

  const cursor =
    canPan || panning.current ? (panning.current ? "grabbing" : "grab") : tool === "drawPolygon" ? "crosshair" : "default";

  return (
    <div ref={wrapRef} className="relative min-h-[280px] min-w-0 flex-1 rounded-xl border border-border bg-viewer-canvas" style={{ cursor }}>
      <Stage
        ref={stageRef}
        width={size.w}
        height={size.h}
        onWheel={handleWheel}
        onMouseDown={onStageMouseDown}
        onMouseMove={(e) => onStageMouseMove(e)}
        onMouseUp={onStageMouseUp}
        onMouseLeave={() => {
          onScenePointer?.(null);
          onStageMouseUp();
        }}
        style={{ borderRadius: "0.75rem" }}
      >
        <Layer x={stagePosition.x} y={stagePosition.y} scaleX={stageScale} scaleY={stageScale}>
          {canvasImage ? (
            <KonvaImage
              ref={imageRef}
              image={canvasImage}
              x={0}
              y={0}
              width={imgW}
              height={imgH}
              listening={false}
            />
          ) : null}

          {gridLines && canvasImage ? (
            <Line
              points={[0, 0, imgW, 0, imgW, imgH, 0, imgH, 0, 0]}
              stroke="rgba(148,163,184,0.15)"
              strokeWidth={1 / stageScale}
              listening={false}
            />
          ) : null}

          {visibleFindings.map((f) => (
            <Line
              key={f.id}
              points={f.polygon.flatMap((p) => [p.x, p.y])}
              closed
              fill={fillForFinding(f)}
              stroke={selectedId === f.id ? "#f8fafc" : hoveredId === f.id ? "#e2e8f0" : strokeForFinding(f)}
              strokeWidth={(selectedId === f.id ? 3.5 : hoveredId === f.id ? 2.8 : 2) / stageScale}
              shadowBlur={selectedId === f.id || hoveredId === f.id ? 10 / stageScale : 0}
              shadowColor="rgba(59,130,246,0.6)"
              onClick={(e) => {
                e.cancelBubble = true;
                if (tool === "erase") {
                  onEraseFinding?.(f.id);
                  return;
                }
                onSelectId(f.id);
              }}
              draggable={tool === "select" && selectedId === f.id}
              onDragStart={(e) => {
                e.cancelBubble = true;
                onPolygonEditStart?.(f.id);
                const st = e.target.getStage();
                const p = st?.getPointerPosition();
                dragPoly.current = { id: f.id, last: p ? toScene(p) : null };
              }}
              onDragMove={(e) => {
                const st = e.target.getStage();
                const p = st?.getPointerPosition();
                if (!p || !dragPoly.current || dragPoly.current.id !== f.id) return;
                const cur = toScene(p);
                const last = dragPoly.current.last;
                if (!last) {
                  dragPoly.current.last = cur;
                  return;
                }
                const fac = precisionMode ? 0.35 : 1;
                const dx = (cur.x - last.x) * fac;
                const dy = (cur.y - last.y) * fac;
                onPolygonChange(
                  f.id,
                  f.polygon.map((pt) => ({ x: pt.x + dx, y: pt.y + dy })),
                );
                dragPoly.current.last = cur;
                e.target.position({ x: 0, y: 0 });
              }}
              onDragEnd={(e) => {
                e.cancelBubble = true;
                e.target.position({ x: 0, y: 0 });
                dragPoly.current = null;
                onPolygonCommit(f.id);
              }}
            />
          ))}

          {selectedFinding &&
            tool === "select" &&
            selectedFinding.polygon.map((point, idx) => {
              const nextIdx = (idx + 1) % selectedFinding.polygon.length;
              const nextPoint = selectedFinding.polygon[nextIdx];
              const midX = (point.x + nextPoint.x) / 2;
              const midY = (point.y + nextPoint.y) / 2;
              return (
                <Circle
                  key={`edge-${selectedFinding.id}-${idx}`}
                  x={midX}
                  y={midY}
                  radius={5 / stageScale}
                  fill="#f97316"
                  stroke="#fff"
                  strokeWidth={1 / stageScale}
                  draggable
                  onDragStart={(e) => {
                    e.cancelBubble = true;
                    onPolygonEditStart?.(selectedFinding.id);
                    const st = e.target.getStage();
                    const p = st?.getPointerPosition();
                    edgeLast.current = p ? toScene(p) : null;
                  }}
                  onDragMove={(e) => {
                    const st = e.target.getStage();
                    const p = st?.getPointerPosition();
                    if (!p) return;
                    const scene = toScene(p);
                    const prev = edgeLast.current;
                    if (!prev) {
                      edgeLast.current = scene;
                      return;
                    }
                    const fac = precisionMode ? 0.35 : 1;
                    let dx = (scene.x - prev.x) * fac;
                    let dy = (scene.y - prev.y) * fac;
                    if (e.evt.shiftKey) {
                      if (Math.abs(dx) > Math.abs(dy)) dy = 0;
                      else dx = 0;
                    }
                    const poly = [...selectedFinding.polygon];
                    poly[idx] = { x: poly[idx].x + dx, y: poly[idx].y + dy };
                    poly[nextIdx] = { x: poly[nextIdx].x + dx, y: poly[nextIdx].y + dy };
                    onPolygonChange(selectedFinding.id, poly);
                    edgeLast.current = scene;
                    e.target.position({ x: midX, y: midY });
                  }}
                  onDragEnd={(e) => {
                    e.cancelBubble = true;
                    edgeLast.current = null;
                    onPolygonCommit(selectedFinding.id);
                  }}
                  onDblClick={(e) => {
                    e.cancelBubble = true;
                    const mid = { x: midX, y: midY };
                    onInsertVertex?.(selectedFinding.id, idx, mid);
                  }}
                />
              );
            })}

          {selectedFinding &&
            tool === "select" &&
            selectedFinding.polygon.map((point, idx) => (
              <Circle
                key={`v-${selectedFinding.id}-${idx}`}
                x={point.x}
                y={point.y}
                radius={7 / stageScale}
                fill="#0f172a"
                stroke="#f8fafc"
                strokeWidth={1.5 / stageScale}
                draggable
                onDragStart={(e) => {
                  e.cancelBubble = true;
                  onPolygonEditStart?.(selectedFinding.id);
                }}
                onDragMove={(e) => {
                  const st = e.target.getStage();
                  const p = st?.getPointerPosition();
                  if (!p) return;
                  let scene = toScene(p);
                  if (e.evt.shiftKey) {
                    const ox = scene.x - point.x;
                    const oy = scene.y - point.y;
                    if (Math.abs(ox) > Math.abs(oy)) scene = { x: scene.x, y: point.y };
                    else scene = { x: point.x, y: scene.y };
                  }
                  const poly = [...selectedFinding.polygon];
                  poly[idx] = scene;
                  onPolygonChange(selectedFinding.id, poly);
                  e.target.position({ x: scene.x, y: scene.y });
                }}
                onDragEnd={(e) => {
                  e.cancelBubble = true;
                  onPolygonCommit(selectedFinding.id);
                }}
                onContextMenu={(e) => {
                  e.evt.preventDefault();
                  if (selectedFinding.polygon.length <= 3) return;
                  onDeleteVertex?.(selectedFinding.id, idx);
                }}
              />
            ))}

          {tool === "drawPolygon" && draftPolygon.length > 0 ? (
            <Line
              points={draftPolygon.flatMap((p) => [p.x, p.y])}
              stroke="#f97316"
              strokeWidth={2 / stageScale}
              dash={[8 / stageScale, 5 / stageScale]}
              listening={false}
            />
          ) : null}

          {boxDraft && boxDraft.w > 0 ? (
            <Line
              points={[boxDraft.x, boxDraft.y, boxDraft.x + boxDraft.w, boxDraft.y, boxDraft.x + boxDraft.w, boxDraft.y + boxDraft.h, boxDraft.x, boxDraft.y + boxDraft.h, boxDraft.x, boxDraft.y]}
              stroke="#f97316"
              strokeWidth={2 / stageScale}
              dash={[6 / stageScale, 4 / stageScale]}
              listening={false}
            />
          ) : null}
        </Layer>
      </Stage>
    </div>
  );
}
