"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, getHealth } from "@/lib/api";
import type { ServiceStatus } from "@/lib/types";
import { Button, Card, Notice, PageHeader, StatusBadge } from "@/components/ui";

export default function HealthPage() {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [detailed, setDetailed] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getHealth();
      setServices(res.services);
      setDetailed(res.detailed);
      setError(null);
    } catch (e) {
      setServices([]);
      setError(e instanceof ApiError ? e : new ApiError("Health check failed."));
    } finally {
      setCheckedAt(new Date());
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load on mount
    refresh();
  }, [refresh]);

  return (
    <>
      <PageHeader
        title="System health"
        subtitle="Status of the API, vector database, models and pipeline services."
        actions={
          <Button variant="ghost" onClick={refresh} disabled={loading}>
            {loading ? "Checking..." : "Re-check"}
          </Button>
        }
      />

      {error ? (
        <Notice tone="error" title={error.message}>
          {error.hint}
        </Notice>
      ) : null}

      {!error && !detailed ? (
        <div className="mb-4">
          <Notice tone="warn" title="Only the basic API check is available">
            The backend has no dependency check yet, so Ollama, Chroma, the models, Airflow and
            MinIO cannot be shown. See the API contract for <code>/health/dependencies</code>.
          </Notice>
        </div>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {services.map((s) => (
          <Card key={s.name}>
            <div className="flex items-center justify-between gap-3">
              <h2 className="font-medium">{s.name}</h2>
              <StatusBadge state={s.status} />
            </div>
            {s.detail ? <p className="mt-2 text-sm text-mute">{s.detail}</p> : null}
          </Card>
        ))}
      </div>

      {checkedAt ? (
        <p className="mt-4 text-xs text-mute">Last checked {checkedAt.toLocaleTimeString()}</p>
      ) : null}
    </>
  );
}
