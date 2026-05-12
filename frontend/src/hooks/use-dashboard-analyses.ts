"use client";

import { useCallback, useEffect, useState } from "react";

import type { AnalysisListItem } from "@/lib/api-types";
import { listAnalyses } from "@/lib/api-client";
import { AUTH_CHANGED_EVENT } from "@/lib/auth";

export function useDashboardAnalyses(limit = 100) {
  const [analyses, setAnalyses] = useState<AnalysisListItem[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setAnalyses(await listAnalyses(limit));
    } catch {
      setAnalyses([]);
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const onAuth = () => void load();
    window.addEventListener(AUTH_CHANGED_EVENT, onAuth);
    return () => window.removeEventListener(AUTH_CHANGED_EVENT, onAuth);
  }, [load]);

  return { analyses, loading, reload: load };
}
