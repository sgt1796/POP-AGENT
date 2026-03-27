"use client";

import type { AssistantMessageProps } from "@copilotkit/react-ui";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";

import { PopUiRenderer } from "@/components/pop-ui-renderer";
import type { UISpec } from "@/lib/types";

type WorkspaceZoneId = string;
type DockPlacement = WorkspaceZoneId | "inline";

export type GeneratedUiEntry = {
  id: string;
  spec: UISpec;
  sourceText: string;
};

export type WorkspacePanelSpec = {
  id: string;
  title: string;
  content: ReactNode;
};

export type WorkspaceColumnSpec = {
  id: string;
  title: string;
  panels: WorkspacePanelSpec[];
};

type DockedUiEntry = {
  entryId: string;
  spec: UISpec;
};

type DragState = {
  entryId: string;
  pointerX: number;
  pointerY: number;
};

type GeneratedUiDockContextValue = {
  uiEntries: GeneratedUiEntry[];
  zoneLabels: Record<WorkspaceZoneId, string>;
  dockedUiByEntryId: Partial<Record<string, WorkspaceZoneId>>;
  dockedUiByZoneId: Record<WorkspaceZoneId, DockedUiEntry[]>;
  dragState: DragState | null;
  hoveredPlacement: WorkspaceZoneId | null;
  setDockPlacement: (entryId: string, placement: DockPlacement) => void;
  startDragging: (entryId: string, pointerX: number, pointerY: number) => void;
  setHoveredPlacement: (placement: WorkspaceZoneId | null) => void;
};

const GeneratedUiDockContext = createContext<GeneratedUiDockContextValue | null>(null);

function useGeneratedUiDock() {
  const context = useContext(GeneratedUiDockContext);
  if (!context) {
    throw new Error("Generated UI dock context is unavailable.");
  }
  return context;
}

function normalizeUiSourceText(value: string | undefined) {
  return String(value || "").replace(/\r\n?/g, "\n").trim();
}

function getColumnStartZoneId(columnId: string) {
  return `${columnId}:start`;
}

function getAfterPanelZoneId(columnId: string, panelId: string) {
  return `${columnId}:after:${panelId}`;
}

function buildZoneLabelForColumnStart(column: WorkspaceColumnSpec) {
  return column.panels[0] ? `above ${column.panels[0].title}` : `at the top of ${column.title}`;
}

function buildZoneLabelAfterPanel(panel: WorkspacePanelSpec, nextPanel: WorkspacePanelSpec | undefined) {
  return nextPanel ? `between ${panel.title} and ${nextPanel.title}` : `below ${panel.title}`;
}

function buildDockGroups(
  uiEntries: GeneratedUiEntry[],
  dockedUiByEntryId: Partial<Record<string, WorkspaceZoneId>>
) {
  const entryById = new Map(uiEntries.map((entry) => [entry.id, entry]));
  const groups: Record<WorkspaceZoneId, DockedUiEntry[]> = {};

  for (const [entryId, zoneId] of Object.entries(dockedUiByEntryId)) {
    if (!zoneId) {
      continue;
    }
    const entry = entryById.get(entryId);
    if (!entry) {
      continue;
    }
    groups[zoneId] ||= [];
    groups[zoneId].push({ entryId: entry.id, spec: entry.spec });
  }

  for (const entries of Object.values(groups)) {
    entries.sort((left, right) => right.entryId.localeCompare(left.entryId));
  }

  return groups;
}

function pointFallsWithinZone(pointerX: number, pointerY: number, node: HTMLElement | null) {
  if (!node) {
    return false;
  }
  const rect = node.getBoundingClientRect();
  return pointerX >= rect.left && pointerX <= rect.right && pointerY >= rect.top && pointerY <= rect.bottom;
}

function formatDropLabel(label: string) {
  return `Drop ${label}`;
}

function fallbackZoneForColumn(
  missingZoneId: WorkspaceZoneId,
  zoneLabels: Record<WorkspaceZoneId, string>,
  zoneOrder: WorkspaceZoneId[]
) {
  const columnId = missingZoneId.split(":")[0];
  const candidates = zoneOrder.filter((zoneId) => zoneId.startsWith(`${columnId}:`) && zoneLabels[zoneId]);
  return candidates.at(-1) || zoneOrder.find((zoneId) => zoneLabels[zoneId]) || null;
}

function buildMessageAssignmentMap(
  messages: AssistantMessageProps["messages"],
  uiEntries: GeneratedUiEntry[]
) {
  const assistantMessages = (messages || []).filter(
    (message): message is NonNullable<AssistantMessageProps["message"]> =>
      message?.role === "assistant" && typeof message.id === "string"
  );

  const uiEntriesByText = new Map<string, GeneratedUiEntry[]>();
  for (const entry of uiEntries) {
    const normalizedText = normalizeUiSourceText(entry.sourceText);
    if (!normalizedText) {
      continue;
    }
    const existing = uiEntriesByText.get(normalizedText);
    if (existing) {
      existing.push(entry);
    } else {
      uiEntriesByText.set(normalizedText, [entry]);
    }
  }

  const entryIdByMessageId: Record<string, string> = {};
  for (const [normalizedText, entriesForText] of uiEntriesByText.entries()) {
    const matchingMessages = assistantMessages.filter(
      (message) => normalizeUiSourceText(message.content) === normalizedText
    );
    const matchCount = Math.min(entriesForText.length, matchingMessages.length);
    for (let offset = 0; offset < matchCount; offset += 1) {
      const entry = entriesForText[entriesForText.length - 1 - offset];
      const message = matchingMessages[matchingMessages.length - 1 - offset];
      entryIdByMessageId[message.id] = entry.id;
    }
  }

  return entryIdByMessageId;
}

function resolveUiEntryForMessage(
  messageId: string | undefined,
  messages: AssistantMessageProps["messages"],
  uiEntries: GeneratedUiEntry[]
) {
  if (!messageId) {
    return null;
  }
  const entryIdByMessageId = buildMessageAssignmentMap(messages, uiEntries);
  const entryId = entryIdByMessageId[messageId];
  if (!entryId) {
    return null;
  }
  return uiEntries.find((entry) => entry.id === entryId) || null;
}

function GeneratedUiCardShell({
  entryId,
  spec,
  placement,
}: {
  entryId: string;
  spec: UISpec;
  placement: DockPlacement;
}) {
  const { dragState, setDockPlacement, startDragging } = useGeneratedUiDock();
  const [menuOpen, setMenuOpen] = useState(false);
  const isDragging = dragState?.entryId === entryId;
  const isInline = placement === "inline";

  useEffect(() => {
    if (!menuOpen) {
      return;
    }

    const handleWindowPointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        setMenuOpen(false);
        return;
      }
      if (!target.closest(`[data-generated-ui-menu="${entryId}"]`)) {
        setMenuOpen(false);
      }
    };

    window.addEventListener("pointerdown", handleWindowPointerDown);
    return () => window.removeEventListener("pointerdown", handleWindowPointerDown);
  }, [entryId, menuOpen]);

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    setMenuOpen(false);
    startDragging(entryId, event.clientX, event.clientY);
  };

  return (
    <div className={`generated-ui-shell ${isInline ? "placement-inline" : "placement-docked"}${isDragging ? " is-dragging" : ""}`}>
      <div className="generated-ui-toolbar">
        <button
          type="button"
          className="generated-ui-grab"
          onPointerDown={handlePointerDown}
          title="Drag this generated UI anywhere in the workspace."
        >
          <span className="generated-ui-grab-dots" aria-hidden="true">
            ::::
          </span>
          <span className="generated-ui-toolbar-label">Generated UI</span>
        </button>

        <div className={`generated-ui-menu${menuOpen ? " open" : ""}`} data-generated-ui-menu={entryId}>
          <button
            type="button"
            className="generated-ui-menu-trigger"
            aria-label="Generated UI placement options"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((current) => !current)}
          >
            ...
          </button>
          {menuOpen ? (
            <div className="generated-ui-menu-list" role="menu">
              {isInline ? (
                <div className="generated-ui-menu-hint">Drag this card to place it anywhere on the page.</div>
              ) : (
                <button
                  type="button"
                  role="menuitem"
                  className="generated-ui-menu-item"
                  onClick={() => {
                    setMenuOpen(false);
                    setDockPlacement(entryId, "inline");
                  }}
                >
                  Move Inline
                </button>
              )}
            </div>
          ) : null}
        </div>
      </div>
      <PopUiRenderer spec={spec} className="generated-ui-card" />
    </div>
  );
}

function GeneratedUiDockStack({ entries }: { entries: DockedUiEntry[] }) {
  if (entries.length === 0) {
    return null;
  }

  return (
    <section className="generated-ui-dock-stack">
      {entries.map((entry) => (
        <GeneratedUiCardShell
          key={`docked-${entry.entryId}`}
          entryId={entry.entryId}
          spec={entry.spec}
          placement="docked"
        />
      ))}
    </section>
  );
}

function GeneratedUiDropSlot({ zoneId, label }: { zoneId: WorkspaceZoneId; label: string }) {
  const { dockedUiByZoneId, dragState, hoveredPlacement } = useGeneratedUiDock();
  const entries = dockedUiByZoneId[zoneId] || [];

  if (!dragState && entries.length === 0) {
    return null;
  }

  return (
    <section className="generated-ui-slot">
      {dragState ? (
        <div
          data-generated-ui-dropzone={zoneId}
          className={`generated-ui-dropzone generated-ui-dropzone-flow${
            hoveredPlacement === zoneId ? " active" : ""
          }`}
          aria-hidden="true"
        >
          <span>{formatDropLabel(label)}</span>
        </div>
      ) : null}
      <GeneratedUiDockStack entries={entries} />
    </section>
  );
}

function GeneratedUiDragPreview() {
  const { uiEntries, dragState } = useGeneratedUiDock();

  if (!dragState) {
    return null;
  }

  const entry = uiEntries.find((candidate) => candidate.id === dragState.entryId);
  if (!entry) {
    return null;
  }

  const previewStyle = {
    left: dragState.pointerX + 18,
    top: dragState.pointerY + 18,
  };

  return (
    <div className="generated-ui-drag-preview" style={previewStyle} aria-hidden="true">
      <div className="generated-ui-drag-preview-badge">Move UI</div>
      <PopUiRenderer spec={entry.spec} className="generated-ui-card generated-ui-preview-card" />
    </div>
  );
}

export function buildWorkspaceZoneMetadata(columns: WorkspaceColumnSpec[]) {
  const zoneLabels: Record<WorkspaceZoneId, string> = {};
  const zoneOrder: WorkspaceZoneId[] = [];

  for (const column of columns) {
    const startZoneId = getColumnStartZoneId(column.id);
    zoneLabels[startZoneId] = buildZoneLabelForColumnStart(column);
    zoneOrder.push(startZoneId);

    column.panels.forEach((panel, index) => {
      const zoneId = getAfterPanelZoneId(column.id, panel.id);
      zoneLabels[zoneId] = buildZoneLabelAfterPanel(panel, column.panels[index + 1]);
      zoneOrder.push(zoneId);
    });
  }

  return { zoneLabels, zoneOrder };
}

export function GeneratedUiWorkspaceColumn({ id, title, panels }: WorkspaceColumnSpec) {
  return (
    <div className="generated-ui-workspace-column">
      <GeneratedUiDropSlot
        zoneId={getColumnStartZoneId(id)}
        label={buildZoneLabelForColumnStart({ id, title, panels })}
      />
      {panels.map((panel, index) => (
        <div key={panel.id} className="generated-ui-panel-block">
          {panel.content}
          <GeneratedUiDropSlot
            zoneId={getAfterPanelZoneId(id, panel.id)}
            label={buildZoneLabelAfterPanel(panel, panels[index + 1])}
          />
        </div>
      ))}
    </div>
  );
}

export function GeneratedUiDockProvider({
  uiEntries,
  zoneLabels,
  zoneOrder,
  children,
}: {
  uiEntries: GeneratedUiEntry[];
  zoneLabels: Record<WorkspaceZoneId, string>;
  zoneOrder: WorkspaceZoneId[];
  children: ReactNode;
}) {
  const [dockedUiByEntryId, setDockedUiByEntryId] = useState<Partial<Record<string, WorkspaceZoneId>>>({});
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [hoveredPlacement, setHoveredPlacement] = useState<WorkspaceZoneId | null>(null);
  const hoveredPlacementRef = useRef<WorkspaceZoneId | null>(null);

  const dockedUiByZoneId = useMemo(
    () => buildDockGroups(uiEntries, dockedUiByEntryId),
    [uiEntries, dockedUiByEntryId]
  );

  useEffect(() => {
    hoveredPlacementRef.current = hoveredPlacement;
  }, [hoveredPlacement]);

  useEffect(() => {
    setDockedUiByEntryId((current) => {
      let changed = false;
      const next = { ...current };
      const knownEntryIds = new Set(uiEntries.map((entry) => entry.id));

      for (const entryId of Object.keys(next)) {
        if (!knownEntryIds.has(entryId)) {
          delete next[entryId];
          changed = true;
        }
      }

      for (const [entryId, zoneId] of Object.entries(next)) {
        if (!zoneId || zoneLabels[zoneId]) {
          continue;
        }
        const fallbackZoneId = fallbackZoneForColumn(zoneId, zoneLabels, zoneOrder);
        if (fallbackZoneId) {
          next[entryId] = fallbackZoneId;
        } else {
          delete next[entryId];
        }
        changed = true;
      }

      return changed ? next : current;
    });
  }, [uiEntries, zoneLabels, zoneOrder]);

  useEffect(() => {
    if (!dragState) {
      return;
    }

    const resolvePlacement = (pointerX: number, pointerY: number) => {
      const dropzones = document.querySelectorAll<HTMLElement>("[data-generated-ui-dropzone]");
      for (const dropzone of dropzones) {
        if (pointFallsWithinZone(pointerX, pointerY, dropzone)) {
          return dropzone.dataset.generatedUiDropzone || null;
        }
      }
      return null;
    };

    const handlePointerMove = (event: PointerEvent) => {
      setDragState((current) =>
        current
          ? {
              ...current,
              pointerX: event.clientX,
              pointerY: event.clientY,
            }
          : current
      );

      const nextPlacement = resolvePlacement(event.clientX, event.clientY);
      if (hoveredPlacementRef.current !== nextPlacement) {
        hoveredPlacementRef.current = nextPlacement;
        setHoveredPlacement(nextPlacement);
      }
    };

    const handlePointerUp = () => {
      const finalPlacement = hoveredPlacementRef.current;
      if (finalPlacement) {
        setDockedUiByEntryId((current) => ({
          ...current,
          [dragState.entryId]: finalPlacement,
        }));
      }
      setDragState(null);
      setHoveredPlacement(null);
      hoveredPlacementRef.current = null;
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp, { once: true });

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [dragState]);

  const value = useMemo<GeneratedUiDockContextValue>(
    () => ({
      uiEntries,
      zoneLabels,
      dockedUiByEntryId,
      dockedUiByZoneId,
      dragState,
      hoveredPlacement,
      setDockPlacement: (entryId: string, placement: DockPlacement) => {
        setDockedUiByEntryId((current) => {
          const next = { ...current };
          if (placement === "inline") {
            delete next[entryId];
          } else {
            next[entryId] = placement;
          }
          return next;
        });
      },
      startDragging: (entryId: string, pointerX: number, pointerY: number) => {
        setDragState({ entryId, pointerX, pointerY });
        setHoveredPlacement(null);
        hoveredPlacementRef.current = null;
      },
      setHoveredPlacement,
    }),
    [uiEntries, zoneLabels, dockedUiByEntryId, dockedUiByZoneId, dragState, hoveredPlacement]
  );

  return (
    <GeneratedUiDockContext.Provider value={value}>
      {children}
      <GeneratedUiDragPreview />
    </GeneratedUiDockContext.Provider>
  );
}

export function GeneratedUiMessageAttachment({
  messageId,
  messages,
}: {
  messageId: string | undefined;
  messages: AssistantMessageProps["messages"];
}) {
  const { uiEntries, dockedUiByEntryId, zoneLabels } = useGeneratedUiDock();
  const entry = useMemo(() => resolveUiEntryForMessage(messageId, messages, uiEntries), [messageId, messages, uiEntries]);

  if (!entry) {
    return null;
  }

  const zoneId = dockedUiByEntryId[entry.id];
  if (zoneId) {
    return (
      <div className="generated-ui-inline-note">
        Generated UI moved to {zoneLabels[zoneId] || "another place in the workspace"}.
      </div>
    );
  }

  return <GeneratedUiCardShell entryId={entry.id} spec={entry.spec} placement="inline" />;
}
