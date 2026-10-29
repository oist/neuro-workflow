import { describe, expect, it } from "vitest";
import { flowChrome, readableSurface } from "./readableSurface";

describe("readableSurface", () => {
  it("keeps the light panels on the colors they use today", () => {
    expect(readableSurface("light")).toEqual({
      bg: "white",
      text: "gray.800",
      muted: "gray.600",
      border: "gray.200",
      hover: "gray.50",
      tip: "blue.600",
      warning: "orange.600",
      cardBg: "white",
    });
  });

  it("uses a light text color on the dark panels and a darker node card", () => {
    expect(readableSurface("dark")).toEqual({
      bg: "gray.700",
      text: "gray.100",
      muted: "gray.300",
      border: "gray.600",
      hover: "gray.600",
      tip: "blue.300",
      warning: "orange.300",
      cardBg: "gray.800",
    });
  });
});

describe("flowChrome", () => {
  it("keeps the light zoom buttons and minimap", () => {
    const chrome = flowChrome("light");
    expect(chrome.buttonBackground).toBe("white");
    expect(chrome.icon).toBe("#4a5568");
    expect(chrome.minimapBackground).toBe("#f8f9fa");
    expect(chrome.node).toBe("#8b5cf6");
    expect(chrome.css).toContain("background: white");
    expect(chrome.css).toContain("fill: #4a5568");
    expect(chrome.css).toContain("background-color: #f8f9fa");
  });

  it("uses readable dark zoom buttons and keeps the purple minimap nodes", () => {
    const chrome = flowChrome("dark");
    expect(chrome.buttonBackground).toBe("#2D3748");
    expect(chrome.icon).toBe("#EDF2F7");
    expect(chrome.minimapBackground).toBe("#1A202C");
    expect(chrome.mask).toBe("rgba(237, 242, 247, 0.28)");
    expect(chrome.node).toBe("#8b5cf6");
    expect(chrome.css).toContain("fill: #EDF2F7");
    expect(chrome.css).not.toContain("background: white");
  });
});
