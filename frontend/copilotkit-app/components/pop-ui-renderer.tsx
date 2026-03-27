"use client";

import type { UISpec } from "@/lib/types";
import { PlanChecklist, ResultTable, StatGrid } from "@/components/ui-cards";

export function PopUiRenderer({ spec, className }: { spec?: UISpec | null; className?: string }) {
  if (!spec) {
    return null;
  }
  if (spec.type === "StatGrid") {
    return <StatGrid spec={spec} className={className} />;
  }
  if (spec.type === "PlanChecklist") {
    return <PlanChecklist spec={spec} className={className} />;
  }
  if (spec.type === "ResultTable") {
    return <ResultTable spec={spec} className={className} />;
  }
  return null;
}
