export type SurfaceMode = "light" | "dark";

export function readableSurface(mode: SurfaceMode) {
  if (mode === "dark") {
    return {
      bg: "gray.700",
      text: "gray.100",
      muted: "gray.300",
      border: "gray.600",
      hover: "gray.600",
      tip: "blue.300",
      warning: "orange.300",
      cardBg: "gray.800",
    };
  }
  return {
    bg: "white",
    text: "gray.800",
    muted: "gray.600",
    border: "gray.200",
    hover: "gray.50",
    tip: "blue.600",
    warning: "orange.600",
    cardBg: "white",
  };
}

export function toggleIcon(mode: SurfaceMode) {
  return mode === "dark" ? "gray.800" : "gray.600";
}

export function outlineHover(mode: SurfaceMode) {
  if (mode === "dark") {
    return { jupyter: "purple.800", export: "green.800" };
  }
  return { jupyter: "purple.50", export: "green.50" };
}

export function flowChrome(mode: SurfaceMode) {
  const light = mode !== "dark";
  const buttonBackground = light ? "white" : "#2D3748";
  const buttonHoverBackground = light ? "#f7fafc" : "#4A5568";
  const buttonBorder = light ? "#e2e8f0" : "#4A5568";
  const buttonHoverBorder = light ? "#cbd5e0" : "#4A5568";
  const icon = light ? "#4a5568" : "#EDF2F7";
  const minimapBackground = light ? "#f8f9fa" : "#1A202C";
  const minimapBorder = light ? "#e2e8f0" : "#4A5568";
  const mask = light ? "rgba(50, 50, 50, 0.8)" : "rgba(237, 242, 247, 0.28)";
  const node = "#8b5cf6";
  const nodeStroke = "#7c3aed";
  const css = `
    .react-flow__controls {
      background: transparent;
    }
    .react-flow__controls-button {
      background: ${buttonBackground};
      border: 1px solid ${buttonBorder};
      color: ${icon};
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .react-flow__controls-button:hover {
      background: ${buttonHoverBackground};
      border-color: ${buttonHoverBorder};
    }
    .react-flow__controls-button svg {
      fill: ${icon};
    }
    .react-flow__minimap {
      background-color: ${minimapBackground};
      border: 1px solid ${minimapBorder};
    }
    .react-flow__minimap-mask {
      fill: ${mask};
    }
    .react-flow__minimap-node {
      fill: ${node};
      stroke: ${nodeStroke};
    }
  `;
  return {
    buttonBackground,
    buttonHoverBackground,
    buttonBorder,
    icon,
    minimapBackground,
    minimapBorder,
    mask,
    node,
    nodeStroke,
    css,
  };
}
