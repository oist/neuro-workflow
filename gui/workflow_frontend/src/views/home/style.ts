import { MiniMapProps, ControlProps } from "@xyflow/react";
import { flowChrome, SurfaceMode } from "./readableSurface";

export function flowWidgetStyles(mode: SurfaceMode): {
  controls: Partial<ControlProps>;
  minimap: Partial<MiniMapProps>;
} {
  const chrome = flowChrome(mode);
  return {
    controls: {
      style: {
        background: "transparent",
      },
      showZoom: true,
      showFitView: true,
      showInteractive: true,
    },
    minimap: {
      style: {
        background: chrome.minimapBackground,
        border: `1px solid ${chrome.minimapBorder}`,
      },
      maskColor: chrome.mask,
      nodeColor: chrome.node,
    },
  };
}
