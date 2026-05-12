/** Contracts aligned with backend Pydantic schemas. */

export type AnalysisStatus =
  | "Pending AI"
  | "Reviewing"
  | "Reviewed"
  | "Report Generated"
  | "Failed";

export type AlertLevel = "Low" | "Medium" | "High";

export type FindingLayer = "quadrants" | "teeth" | "periapical";

export type FindingSource = "ai" | "manual";

export type AuditActionType = "create" | "update" | "delete" | "review" | "export" | "system";

export type ImageAssetKind =
  | "original"
  | "quadrants_overlay"
  | "quadrants_grid"
  | "teeth_overlay"
  | "periapical_full_overlay"
  | "periapical_quadrants_overlay"
  | "teeth_classification_overlay";

export type AnalysisCreateResponse = {
  id: string;
  status: AnalysisStatus;
};

export type AnalysisListItem = {
  id: string;
  patient_name: string;
  patient_id: string;
  scan_date: string | null;
  status: AnalysisStatus;
  alert_level: AlertLevel | null;
};

export type AnalysisDetail = {
  id: string;
  patient_id: string;
  filename: string;
  scan_date: string | null;
  chief_complaint: string | null;
  status: AnalysisStatus;
  alert_level: AlertLevel | null;
  reviewer: string | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
  findings_count: number;
  image_kinds: string[];
};

export type PolygonPoint = { x: number; y: number };

export type FindingRead = {
  id: string;
  tooth_label: string;
  finding: string;
  confidence: number;
  accepted: boolean;
  polygon: PolygonPoint[];
  layer: FindingLayer;
  source: FindingSource;
  created_at: string;
  updated_at: string;
};

export type FindingCreatePayload = {
  tooth_label: string;
  finding: string;
  confidence: number;
  accepted: boolean;
  polygon: PolygonPoint[];
  layer: FindingLayer;
  source: FindingSource;
};

export type FindingUpdatePayload = {
  tooth_label?: string;
  finding?: string;
  confidence?: number;
  accepted?: boolean;
  polygon?: PolygonPoint[];
  layer?: FindingLayer;
};

export type AuditEntryRead = {
  id: string;
  reviewer: string;
  action: string;
  action_type: AuditActionType;
  target_id: string | null;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  timestamp: string;
};

export type AuditEntryCreatePayload = {
  reviewer: string;
  action: string;
  action_type: AuditActionType;
  target_id?: string | null;
  before?: Record<string, unknown> | null;
  after?: Record<string, unknown> | null;
};

export type ReportDraftPayload = {
  clinical_summary: string;
  impression: string;
  recommendations: string;
  reviewer_confirmation?: string | null;
  include_images: boolean;
};

export type ReportGenerateRequest = ReportDraftPayload & {
  reviewer: string;
};

export type ReportFindingPreview = {
  id: string;
  tooth_label: string;
  finding: string;
  confidence: number;
  layer: string;
};

export type ReportPreviewResponse = {
  analysis_id: string;
  patient_name: string;
  patient_code: string;
  scan_date: string | null;
  accepted_findings_count: number;
  accepted_findings: ReportFindingPreview[];
  sections: Array<{ title: string; body: string }>;
  image_kinds: string[];
};

export type ReportGenerateResponse = {
  report_id: string;
  analysis_id: string;
  status: AnalysisStatus;
  generated_at: string;
  filename: string;
  download_url: string;
};
