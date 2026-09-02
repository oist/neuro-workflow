import React, { useState, useCallback, useRef, useEffect, createContext, useContext } from 'react';
import { Box } from '@chakra-ui/react';
import { Routes, Route } from 'react-router-dom';
import TabBar from './TabBar';
import HomeView from '../../views/home/homeView';
import FileView from '../../views/file/fileListView';
import FileListView from '../../views/file/fileListView';
import CreateFlowPj from '../../views/file/createView';
import BoxUpload from '../../views/box/uploadView';
import NotFoundView from '../../views/notFound/notFound';
import CustomDatabaseManager from '../../views/home/components/CustomDatabaseManager';
import SecretVaultPage from '../../views/home/components/SecretVaultPage';
import UserProfileView from '../../views/user/userProfileView';
import ChatbotArea from '../../views/home/components/chatbotView';
import { useViewerStore, type ViewerSnapshot } from '../../stores/viewerStore';

export interface Tab {
  id: string;
  type: 'workflow' | 'jupyter' | 'viewer';
  title: string;
  projectId?: string;
  url?: string;
  isActive: boolean;
}

interface TabContextType {
  tabs: Tab[];
  activeTabId: string;
  addJupyterTab: (projectId: string, projectName: string, url: string) => void;
  addViewerTab: (tabId: string, title: string, url: string) => void;
  closeTab: (tabId: string) => void;
  switchTab: (tabId: string) => void;
  // Register a viewer tab's iframe so chat actions can postMessage to it.
  registerViewerFrame: (tabId: string, el: HTMLIFrameElement | null) => void;
  // Forward a chat action ({action, ...}) to the relevant viewer iframe.
  // Returns false when no viewer is available to receive it.
  postToActiveViewer: (msg: Record<string, unknown>) => boolean;
}

const TabContext = createContext<TabContextType | undefined>(undefined);

export const useTabContext = () => {
  const context = useContext(TabContext);
  if (!context) {
    throw new Error('useTabContext must be used within TabManager');
  }
  return context;
};

export const TabManager: React.FC = () => {
  const [tabs, setTabs] = useState<Tab[]>([
    {
      id: 'workflow',
      type: 'workflow',
      title: 'Workflow',
      isActive: true,
    }
  ]);
  const [activeTabId, setActiveTabId] = useState('workflow');

  // Viewer iframe registry + latest-snapshot store, for the chat <-> viewer bridge.
  const viewerFrames = useRef<Map<string, HTMLIFrameElement>>(new Map());
  const tabsRef = useRef<Tab[]>(tabs);
  tabsRef.current = tabs;
  const setSnapshot = useViewerStore((s) => s.setSnapshot);
  const clearSnapshot = useViewerStore((s) => s.clearSnapshot);
  const setActiveViewerTabId = useViewerStore((s) => s.setActiveViewerTabId);

  const registerViewerFrame = useCallback((tabId: string, el: HTMLIFrameElement | null) => {
    if (el) viewerFrames.current.set(tabId, el);
    else viewerFrames.current.delete(tabId);
  }, []);

  const postToActiveViewer = useCallback((msg: Record<string, unknown>): boolean => {
    const currentTabs = tabsRef.current;
    const active = currentTabs.find((t) => t.id === activeTabId);
    let targetId: string | null = null;
    if (active?.type === 'viewer') {
      targetId = active.id;
    } else {
      const viewers = currentTabs.filter((t) => t.type === 'viewer');
      if (viewers.length === 1) targetId = viewers[0].id;
      else targetId = useViewerStore.getState().activeViewerTabId;
    }
    const frame = targetId ? viewerFrames.current.get(targetId) : null;
    if (!frame?.contentWindow) return false;
    frame.contentWindow.postMessage({ ...msg, source: 'nw-chat' }, window.location.origin);
    return true;
  }, [activeTabId]);

  // Receive viewer state snapshots (nw-viewer) and store them keyed by tab id.
  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      if (e.origin !== window.location.origin) return;
      const m = e.data;
      if (!m || m.source !== 'nw-viewer' || m.type !== 'snapshot') return;
      for (const [tabId, frame] of viewerFrames.current.entries()) {
        if (frame.contentWindow === e.source) {
          setSnapshot(tabId, m.snapshot as ViewerSnapshot);
          break;
        }
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [setSnapshot]);

  const addJupyterTab = useCallback((projectId: string, projectName: string, url: string) => {
    const existingTab = tabs.find(tab => tab.type === 'jupyter' && tab.id === `${projectId}-${projectName}`);
    
    if (existingTab) {
      // Activate an existing tab
      switchTab(existingTab.id);
      return;
    }

    const newTabId = `${projectId}-${projectName}`;
    const newTab: Tab = {
      id: newTabId,
      type: 'jupyter',
      //title: `${projectName} - JupyterLab`,
      title: `${projectName}`,
      projectId,
      url,
      isActive: true,
    };

    setTabs(prevTabs => {
      const updatedTabs = prevTabs.map(tab => ({ ...tab, isActive: false }));
      return [...updatedTabs, newTab];
    });
    setActiveTabId(newTabId);
  }, [tabs]);

  const addViewerTab = useCallback((tabId: string, title: string, url: string) => {
    setTabs(prevTabs => {
      const exists = prevTabs.some(tab => tab.id === tabId);

      if (exists) {
        // Reload the existing viewer tab with the latest data (url carries a cache-buster)
        return prevTabs.map(tab =>
          tab.id === tabId
            ? { ...tab, url, isActive: true }
            : { ...tab, isActive: false }
        );
      }

      const newTab: Tab = {
        id: tabId,
        type: 'viewer',
        title,
        url,
        isActive: true,
      };
      const updatedTabs = prevTabs.map(tab => ({ ...tab, isActive: false }));
      return [...updatedTabs, newTab];
    });
    setActiveTabId(tabId);
    setActiveViewerTabId(tabId);
  }, [setActiveViewerTabId]);

  const closeTab = useCallback((tabId: string) => {
    if (tabId === 'workflow') return; // Workflow tab cannot be closed

    // Drop the viewer iframe registration + snapshot for a closed viewer tab.
    viewerFrames.current.delete(tabId);
    clearSnapshot(tabId);
    if (useViewerStore.getState().activeViewerTabId === tabId) {
      setActiveViewerTabId(null);
    }

    setTabs(prevTabs => {
      const updatedTabs = prevTabs.filter(tab => tab.id !== tabId);

      // Activate the workflow tab if a closed tab was active
      if (tabId === activeTabId) {
        setActiveTabId('workflow');
        return updatedTabs.map(tab =>
          tab.id === 'workflow' ? { ...tab, isActive: true } : { ...tab, isActive: false }
        );
      }

      return updatedTabs;
    });
  }, [activeTabId, clearSnapshot, setActiveViewerTabId]);

  const switchTab = useCallback((tabId: string) => {
    setTabs(prevTabs =>
      prevTabs.map(tab => ({ ...tab, isActive: tab.id === tabId }))
    );
    setActiveTabId(tabId);
    // Track the last-activated viewer so the chat targets what the user sees.
    if (tabsRef.current.find((t) => t.id === tabId)?.type === 'viewer') {
      setActiveViewerTabId(tabId);
    }
  }, [setActiveViewerTabId]);

  const activeTab = tabs.find(tab => tab.id === activeTabId);

  const contextValue: TabContextType = {
    tabs,
    activeTabId,
    addJupyterTab,
    addViewerTab,
    closeTab,
    switchTab,
    registerViewerFrame,
    postToActiveViewer,
  };

  return (
    <TabContext.Provider value={contextValue}>
      <Box
        flex={1}
        minH={0}
        display="flex"
        flexDirection="column"
      >
        {/* tab bar */}
        <Box flexShrink={0}>
          <TabBar />
        </Box>

        {/* Content area */}
        <Box flex="1" minH="0" overflow="hidden" position="relative">
          {tabs.map(tab => {

            if (tab.type === 'workflow') {
              return (
                <Box
                  key={tab.id}
                  position="absolute"
                  top={0}
                  left={0}
                  right={0}
                  bottom={0}
                  display={tab.id === activeTabId ? "block" : "none"}
                >
                  <Routes>
                    <Route path="/" element={<HomeView />} />
                    <Route path="/file" element={<FileListView />} />
                    <Route path="/file/new" element={<CreateFlowPj />} />
                    <Route path="/box/upload" element={<BoxUpload />} />
                    <Route path="/settings/databases" element={<CustomDatabaseManager />} />
                    <Route path="/settings/secrets" element={<SecretVaultPage />} />
                    <Route path="/user" element={<UserProfileView />} />
                    <Route path="/*" element={<NotFoundView />} />
                  </Routes>
                </Box>
              );
            }

            if (tab.type === 'jupyter' && tab.url) {
              return (
                <Box
                  key={tab.id}
                  position="absolute"
                  top={0}
                  left={0}
                  right={0}
                  bottom={0}
                  display={tab.id === activeTabId ? "block" : "none"}
                >
                  <iframe
                    src={tab.url}
                    width="100%"
                    height="100%"
                    style={{
                      border: 'none',
                      backgroundColor: 'white',
                    }}
                    title={tab.title}
                    sandbox="allow-same-origin allow-scripts allow-forms allow-downloads allow-modals allow-popups allow-popups-to-escape-sandbox"
                  />
                </Box>
              );
            }

            if (tab.type === 'viewer' && tab.url) {
              return (
                <Box
                  key={tab.id}
                  position="absolute"
                  top={0}
                  left={0}
                  right={0}
                  bottom={0}
                  display={tab.id === activeTabId ? "block" : "none"}
                >
                  <iframe
                    ref={(el) => registerViewerFrame(tab.id, el)}
                    src={tab.url}
                    width="100%"
                    height="100%"
                    style={{
                      border: 'none',
                      backgroundColor: 'white',
                    }}
                    title={tab.title}
                    allow="clipboard-write"
                    sandbox="allow-same-origin allow-scripts allow-forms allow-downloads allow-modals allow-popups allow-popups-to-escape-sandbox"
                  />
                </Box>
              );
            }

            return null;
          })}

          {/* Chat lives at the tab-manager level so it overlays every tab —
              notably the brain-viewer tab, whose 3D scene it drives. */}
          <ChatbotArea />
        </Box>

      </Box>
    </TabContext.Provider>
  );
};

export default TabManager;
