import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

type NodeCatalogContextValue = {
  isOpen: boolean;
  initialNodeId: string | null;
  open: (nodeId?: string) => void;
  close: () => void;
};

const NodeCatalogContext = createContext<NodeCatalogContextValue | undefined>(
  undefined
);

export const useNodeCatalog = (): NodeCatalogContextValue => {
  const value = useContext(NodeCatalogContext);
  if (!value) {
    throw new Error("useNodeCatalog must be used within NodeCatalogProvider");
  }
  return value;
};

export const NodeCatalogProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [initialNodeId, setInitialNodeId] = useState<string | null>(null);

  const open = useCallback((nodeId?: string) => {
    setInitialNodeId(nodeId ?? null);
    setIsOpen(true);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  const value = useMemo(
    () => ({ isOpen, initialNodeId, open, close }),
    [isOpen, initialNodeId, open, close]
  );

  return (
    <NodeCatalogContext.Provider value={value}>
      {children}
    </NodeCatalogContext.Provider>
  );
};
