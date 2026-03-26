"use client";

import type { UISpec } from "@/lib/types";
import { PlanChecklist, ResultTable } from "@/components/ui-cards";

export function PopUiRenderer({ spec }: { spec?: UISpec | null }) {
  if (!spec) {
    return null;
  }
  if (spec.type === "PlanChecklist") {
    return <PlanChecklist spec={spec} />;
  }
  if (spec.type === "ResultTable") {
    return <ResultTable spec={spec} />;
  }
  return null;
}
