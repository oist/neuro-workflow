import { describe, expect, it } from "vitest";
import { messageFromProjectCreateError } from "./projectCreateError";

describe("messageFromProjectCreateError", () => {
  it("shows the duplicate-name warning from the name field", () => {
    expect(
      messageFromProjectCreateError(
        { name: ["You already have an active project with this name."] },
        400,
      ),
    ).toBe("You already have an active project with this name.");
  });

  it("keeps a body that only has error", () => {
    expect(messageFromProjectCreateError({ error: "Disk full" }, 400)).toBe(
      "Disk full",
    );
  });
});
