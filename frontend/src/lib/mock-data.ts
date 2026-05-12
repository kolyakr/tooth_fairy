export type AnalysisStatus = "Pending AI" | "Reviewed" | "Report Generated";
export type AlertLevel = "Low" | "Medium" | "High";

export type RecentAnalysis = {
  id: string;
  patientName: string;
  patientId: string;
  scanDate: string;
  status: AnalysisStatus;
  alertLevel: AlertLevel;
};

export type Finding = {
  id: string;
  toothLabel: string;
  finding: string;
  confidence: number;
  accepted: boolean;
  polygon: { x: number; y: number }[];
  layer: "teeth" | "periapical" | "quadrants";
};

export const recentAnalyses: RecentAnalysis[] = [
  {
    id: "A-1001",
    patientName: "Olivia Jensen",
    patientId: "P-4021",
    scanDate: "2026-05-02",
    status: "Reviewed",
    alertLevel: "High",
  },
  {
    id: "A-1002",
    patientName: "Mateo Carter",
    patientId: "P-4022",
    scanDate: "2026-05-03",
    status: "Pending AI",
    alertLevel: "Medium",
  },
  {
    id: "A-1003",
    patientName: "Ava Walker",
    patientId: "P-4023",
    scanDate: "2026-05-04",
    status: "Report Generated",
    alertLevel: "Low",
  },
];

export const mockFindings: Finding[] = [
  {
    id: "F-1",
    toothLabel: "FDI-36",
    finding: "Caries",
    confidence: 0.92,
    accepted: true,
    polygon: [
      { x: 205, y: 220 },
      { x: 292, y: 224 },
      { x: 286, y: 286 },
      { x: 212, y: 290 },
    ],
    layer: "teeth",
  },
  {
    id: "F-2",
    toothLabel: "FDI-47",
    finding: "Periapical Lesion",
    confidence: 0.88,
    accepted: false,
    polygon: [
      { x: 492, y: 265 },
      { x: 570, y: 270 },
      { x: 560, y: 320 },
      { x: 500, y: 322 },
    ],
    layer: "periapical",
  },
  {
    id: "F-3",
    toothLabel: "Q2",
    finding: "Quadrant Region",
    confidence: 0.98,
    accepted: true,
    polygon: [
      { x: 340, y: 120 },
      { x: 620, y: 120 },
      { x: 620, y: 315 },
      { x: 340, y: 315 },
    ],
    layer: "quadrants",
  },
];
