import { useEffect, useState } from "react";
import { Input, NumberInput, NumberInputField } from "@chakra-ui/react";

// Inputs that hold a draft while typing and commit on blur or Enter, so a
// table cell does not fire a request per keystroke.

interface DraftNumberProps {
  value: number | undefined;
  onCommit: (value: number | undefined) => void;
  allowEmpty?: boolean;
  placeholder?: string;
  isInvalid?: boolean;
  w?: string;
}

export const DraftNumberInput = ({
  value, onCommit, allowEmpty = false, placeholder, isInvalid, w = "90px",
}: DraftNumberProps) => {
  const text = value === undefined ? "" : String(value);
  const [draft, setDraft] = useState(text);
  useEffect(() => setDraft(text), [text]);

  const commit = () => {
    if (draft.trim() === "") {
      if (allowEmpty) onCommit(undefined);
      else setDraft(text);
      return;
    }
    const n = Number(draft);
    if (Number.isFinite(n)) {
      if (n !== value) onCommit(n);
    } else {
      setDraft(text);
    }
  };

  return (
    <NumberInput size="sm" value={draft} onChange={(s) => setDraft(s)} w={w} isInvalid={isInvalid}>
      <NumberInputField
        placeholder={placeholder}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
      />
    </NumberInput>
  );
};

interface DraftTextProps {
  value: string;
  onCommit: (value: string) => void;
  placeholder?: string;
  isInvalid?: boolean;
  w?: string;
}

export const DraftTextInput = ({ value, onCommit, placeholder, isInvalid, w = "90px" }: DraftTextProps) => {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = () => {
    const v = draft.trim();
    if (v !== value) onCommit(v);
  };
  return (
    <Input
      size="sm"
      w={w}
      value={draft}
      placeholder={placeholder}
      isInvalid={isInvalid}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
    />
  );
};
