import React, { useState, useEffect, useRef } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Button,
  Box,
  VStack,
  HStack,
  Text,
  useToast,
  Spinner,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Code,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  IconButton,
  Tooltip,
  Input,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverArrow,
  PopoverBody,
} from '@chakra-ui/react';
import { ArrowRightIcon, DownloadIcon, CopyIcon } from '@chakra-ui/icons';
import { createAuthHeaders } from '../../../api/authHeaders';
import { API_BASE_URL } from '../../../config/urls';

// Monaco Editor Type Definitions
interface Monaco {
  editor: {
    create: (container: HTMLElement, options: any) => any;
    defineTheme: (themeName: string, themeData: any) => void;
    setTheme: (themeName: string) => void;
  };
  languages: {
    register: (language: { id: string }) => void;
    setMonarchTokensProvider: (languageId: string, provider: any) => void;
  };
}

// Execution result type definition
interface ExecutionResult {
  status: 'success' | 'error' | 'running';
  output?: string;
  error?: string;
  execution_time?: number;
  timestamp?: string;
}

// Endpoint configuration type definition
interface EndpointConfig {
  baseUrl?: string;
  getCode: string;    // GET: Code retrieval endpoint
  saveCode: string;   // PUT: Code save endpoint
  executeCode?: string; // POST: Code execution endpoint (optional)
  forkCode?: string;    // POST: Save as the user's own copy; {file_id} is the id from the GET (optional)
  revertCode?: string;  // DELETE: Drop a project copy and go back to the library code (optional)
}

// Nodes whose schema the server rebuilt after a save (project copies)
export interface NodeRefreshResult {
  updated: string[];
  dropped_parameters: Record<string, string[]>;
  removed_ports: Record<string, string[]>;
  dangling_edges: string[];
}

// What was saved, passed to onSaved
export interface SavedCode {
  fileId?: string;
  fileName?: string;
  className?: string;
  forked: boolean; // true when saved as a new copy instead of in place
  refresh?: NodeRefreshResult; // set when the server already updated the nodes
}

interface CodeEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  identifier: string | null; // projectId, workflowId, filename General-purpose identifiers such as
  endpoints: EndpointConfig; // Endpoint settings
  title?: string; // Modal title (optional)
  initialCode?: string; // Initial Code (Optional)
  language?: string; // Programming Language (default: python）
  downloadFileName?: string; // Download filename (optional)
  showExecute?: boolean; // Show execution functions (default: true）
  nodeClassName?: string; // Node class the code must keep defining (node files only)
  onSaved?: (saved: SavedCode) => void; // Called after a successful save or copy
}

export const CodeEditorModal: React.FC<CodeEditorModalProps> = ({
  isOpen,
  onClose,
  identifier,
  endpoints,
  title,
  initialCode = '# Generated Python code will appear here\nprint("Hello, World!")',
  language = 'python',
  downloadFileName,
  showExecute = true,
  nodeClassName,
  onSaved,
}) => {
  const toast = useToast();
  const editorRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const monacoRef = useRef<Monaco | null>(null);

  const [code, setCode] = useState<string>(initialCode);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState<boolean>(false);
  // Returned by the GET: whether the user may save in place, and what was loaded
  const [canEdit, setCanEdit] = useState<boolean>(true);
  const [contentHash, setContentHash] = useState<string | undefined>(undefined);
  const [fileId, setFileId] = useState<string | undefined>(undefined);
  const [isConflict, setIsConflict] = useState<boolean>(false);
  const [copyClassName, setCopyClassName] = useState<string>('');
  // 'project' / 'library' when the endpoints serve a project copy of a node
  const [source, setSource] = useState<string | undefined>(undefined);
  const [warnings, setWarnings] = useState<string[]>([]);

  // Get base URL
  const getBaseUrl = () => endpoints.baseUrl || API_BASE_URL;

  // Build endpoint URL
  const buildUrl = (endpoint: string) => {
    const baseUrl = getBaseUrl();
    // Replace if identifier exists, use as is if not
    if (identifier) {
      return `${baseUrl}${endpoint.replace('{identifier}', identifier)}`;
    }
    return `${baseUrl}${endpoint}`;
  };

  // Dynamically Loading the Monaco Editor
  useEffect(() => {
    if (!isOpen) return;

    const loadMonaco = async () => {
      try {
        console.log('🔧 Loading Monaco Editor...');
        
        // Loading Monaco Editor from CDN
        if (!window.monaco) {
          console.log('📦 Monaco not found, loading from CDN...');
          await loadScript('https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs/loader.min.js');
          
          return new Promise<void>((resolve) => {
            window.require.config({ 
              paths: { 
                vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' 
              } 
            });
            
            window.require(['vs/editor/editor.main'], () => {
              console.log('✅ Monaco Editor loaded successfully');
              monacoRef.current = window.monaco;
              resolve();
            });
          });
        } else {
          console.log('✅ Monaco Editor already available');
          monacoRef.current = window.monaco;
        }
      } catch (error) {
        console.error('❌ Failed to load Monaco Editor:', error);
        toast({
          title: "Editor Error",
          description: "Failed to load code editor",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
      }
    };

    loadMonaco();
  }, [isOpen, toast]);

  // Initialize Monaco Editor
  useEffect(() => {
    // Wait until all the conditions are met
    if (!monacoRef.current || !containerRef.current || !isOpen || isLoading) {
      console.log('⏳ Monaco Editor not ready:', {
        monaco: !!monacoRef.current,
        container: !!containerRef.current,
        isOpen,
        isLoading
      });
      return;
    }

    console.log('🎨 All conditions met, initializing Monaco Editor');

    // Destroy the existing editor
    if (editorRef.current) {
      console.log('🗑️ Disposing existing editor');
      editorRef.current.dispose();
      editorRef.current = null;
    }

    // Adding an additional delay to wait for the DOM element to be fully ready
    const initTimer = setTimeout(() => {
      if (!containerRef.current) {
        console.log('❌ Container disappeared during initialization');
        return;
      }

      try {
        console.log('🚀 Starting Monaco Editor creation...');
        
        // Define a custom theme
        monacoRef.current.editor.defineTheme('chakra-theme', {
          base: 'vs',
          inherit: true,
          rules: [
            { token: 'comment', foreground: '6a737d' },
            { token: 'keyword', foreground: 'd73a49' },
            { token: 'string', foreground: '032f62' },
            { token: 'number', foreground: '005cc5' },
          ],
          colors: {
            'editor.background': '#ffffff',
            'editor.foreground': '#24292e',
            'editorLineNumber.foreground': '#959da5',
            'editorCursor.foreground': '#044289',
            'editor.selectionBackground': '#c8e1ff',
            'editor.lineHighlightBackground': '#f6f8fa',
          },
        });

        // Create an editor (initial value is empty)
        editorRef.current = monacoRef.current.editor.create(containerRef.current, {
          value: '',
          language: language,
          theme: 'chakra-theme',
          fontSize: 14,
          lineNumbers: 'on',
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          automaticLayout: true,
          tabSize: 4,
          insertSpaces: true,
          renderWhitespace: 'selection',
          bracketPairColorization: { enabled: true },
        });

        console.log('✅ Monaco Editor created successfully');

        // Monitor code changes
        const disposable = editorRef.current.onDidChangeModelContent(() => {
          const newCode = editorRef.current.getValue();
          setCode(newCode);
          setHasUnsavedChanges(true);
        });

        // Set initial code
        if (code) {
          editorRef.current.setValue(code);
          console.log('📝 Initial code set, length:', code.length);
        }

        // Save the cleanup function
        return () => {
          console.log('🧹 Cleaning up Monaco Editor');
          disposable?.dispose();
          if (editorRef.current) {
            editorRef.current.dispose();
            editorRef.current = null;
          }
        };
      } catch (error) {
        console.error('❌ Error creating Monaco Editor:', error);
      }
    }, 300);

    return () => {
      clearTimeout(initTimer);
    };

  }, [monacoRef.current, isOpen, isLoading, language]); // Remove code from the dependency array

  // Mark (or clear, with no line) the line a save was rejected for
  const showProblemLine = (line?: number, message?: string) => {
    const model = editorRef.current?.getModel?.();
    if (!model || !window.monaco) return;
    window.monaco.editor.setModelMarkers(model, 'save', line ? [{
      startLineNumber: line,
      endLineNumber: line,
      startColumn: 1,
      endColumn: model.getLineMaxColumn(line),
      message: message || 'Error',
      severity: window.monaco.MarkerSeverity.Error,
    }] : []);
    if (line) editorRef.current.revealLineInCenter(line);
  };

  // Report a rejected save, marking the offending line when there is one
  const reportSaveError = async (response: Response) => {
    const body = await response.json().catch(() => ({}));
    showProblemLine(body.line, body.error);
    toast({
      title: "Not saved",
      description: body.error || `Failed to save code (${response.status})`,
      status: "error",
      duration: 6000,
      isClosable: true,
    });
  };

  // get code
  const fetchCode = async () => {
    if (!identifier) return;

    setIsLoading(true);
    try {
      const header = await createAuthHeaders();
      const url = buildUrl(endpoints.getCode);
      
      const response = await fetch(url, {
        credentials: 'include',
        headers: { ...header },
      });

      if (response.ok) {
        const data = await response.json();
        const fetchedCode = data.code || data.file_content || data.source_code || initialCode;
        setCode(fetchedCode);
        setCanEdit(data.can_edit !== false);
        setContentHash(data.content_hash);
        setFileId(data.file_id);
        setSource(data.source);
        setWarnings(data.warnings || []);
        setIsConflict(false);
        setCopyClassName(nodeClassName ? `${nodeClassName}Custom` : '');
        showProblemLine(undefined);
        
        // If the editor is initialized, update the value
        if (editorRef.current) {
          editorRef.current.setValue(fetchedCode);
        }
        
        setHasUnsavedChanges(false);
        
        toast({
          title: "Code Loaded",
          description: "Code loaded successfully",
          status: "success",
          duration: 2000,
          isClosable: true,
        });
      } else {
        throw new Error('Failed to fetch code');
      }
    } catch (error) {
      console.error('Failed to fetch code:', error);
      toast({
        title: "Error",
        description: "Failed to load code",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // save code
  const saveCode = async () => {
    if (!identifier) return;

    setIsSaving(true);
    try {
      const header = await createAuthHeaders();
      const url = buildUrl(endpoints.saveCode);
      
      const response = await fetch(url, {
        method: 'PUT',
        credentials: 'include',
        headers: {
          ...header,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code,
          base_hash: contentHash,
          expected_class: nodeClassName,
        }),
      });

      if (response.ok) {
        const data = await response.json().catch(() => ({}));
        setHasUnsavedChanges(false);
        setContentHash(data.content_hash);
        showProblemLine(undefined);
        toast({
          title: "Saved",
          description: "Code saved successfully",
          status: "success",
          duration: 2000,
          isClosable: true,
        });
        if (data.source) setSource(data.source);
        onSaved?.({
          fileId: data.file_id,
          fileName: data.filename,
          className: nodeClassName,
          forked: false,
          refresh: data.updated ? data : undefined,
        });
      } else {
        // 409: the file changed (e.g. in JupyterLab) after it was loaded
        setIsConflict(response.status === 409);
        await reportSaveError(response);
      }
    } catch (error) {
      console.error('Failed to save code:', error);
      toast({
        title: "Error",
        description: "Failed to save code",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsSaving(false);
    }
  };

  // Save the edited code as the user's own node (a new class and file)
  const saveAsCopy = async () => {
    if (!endpoints.forkCode || !fileId || !nodeClassName) return;

    setIsSaving(true);
    try {
      const header = await createAuthHeaders();
      const url = `${getBaseUrl()}${endpoints.forkCode.replace('{file_id}', fileId)}`;
      const response = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...header,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code,
          class_name: nodeClassName,
          new_class_name: copyClassName.trim(),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setHasUnsavedChanges(false);
        showProblemLine(undefined);
        toast({
          title: "Saved as your node",
          description: `Created ${data.file_name}`,
          status: "success",
          duration: 3000,
          isClosable: true,
        });
        onSaved?.({
          fileId: data.file_id,
          fileName: data.file_name,
          className: data.class_name,
          forked: true,
        });
      } else {
        await reportSaveError(response);
      }
    } catch (error) {
      console.error('Failed to save a copy:', error);
      toast({
        title: "Error",
        description: "Failed to save a copy",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsSaving(false);
    }
  };

  const canSaveCopy = !canEdit && !!endpoints.forkCode && !!nodeClassName;

  // Drop the project copy; the project goes back to the library code
  const revertToLibrary = async () => {
    if (!endpoints.revertCode) return;

    setIsSaving(true);
    try {
      const header = await createAuthHeaders();
      const response = await fetch(buildUrl(endpoints.revertCode), {
        method: 'DELETE',
        credentials: 'include',
        headers: { ...header },
      });
      if (response.ok) {
        const data = await response.json();
        toast({
          title: "Back to the library code",
          description: "The project copy was removed",
          status: "success",
          duration: 3000,
          isClosable: true,
        });
        onSaved?.({ className: nodeClassName, forked: false, refresh: data });
        await fetchCode();
      } else {
        await reportSaveError(response);
      }
    } catch (error) {
      console.error('Failed to remove the project copy:', error);
      toast({
        title: "Error",
        description: "Failed to remove the project copy",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsSaving(false);
    }
  };

  // run code
  const executeCode = async () => {
    if (!identifier || !endpoints.executeCode) return;

    setIsExecuting(true);
    const startTime = Date.now();
    
    try {
      const header = await createAuthHeaders();
      const url = buildUrl(endpoints.executeCode);
      
      const response = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...header,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });

      const data = await response.json();
      const executionTime = data.execution_time || (Date.now() - startTime);

      const result: ExecutionResult = {
        status: response.ok ? 'success' : 'error',
        output: data.output,
        error: data.error,
        execution_time: executionTime,
        timestamp: new Date().toLocaleTimeString(),
      };

      setExecutionResults(prev => [...prev, result]);

      if (response.ok) {
        toast({
          title: "Execution Complete",
          description: `Code executed in ${executionTime}ms`,
          status: "success",
          duration: 2000,
          isClosable: true,
        });
      } else {
        toast({
          title: "Execution Failed",
          description: data.error || "Code execution failed",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (error) {
      console.error('Failed to execute code:', error);
      const result: ExecutionResult = {
        status: 'error',
        error: 'Network error: Failed to execute code',
        execution_time: Date.now() - startTime,
        timestamp: new Date().toLocaleTimeString(),
      };
      setExecutionResults(prev => [...prev, result]);
      
      toast({
        title: "Execution Error",
        description: "Failed to execute code",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsExecuting(false);
    }
  };

  // Copy code to clipboard
  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(code);
      toast({
        title: "Copied",
        description: "Code copied to clipboard",
        status: "success",
        duration: 1000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy code",
        status: "error",
        duration: 2000,
        isClosable: true,
      });
    }
  };

  // download code
  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    
    // Decide file name
    const fileName = downloadFileName || 
                    `${identifier}_code.${language === 'python' ? 'py' : language}`;
    a.download = fileName;
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Downloaded",
      description: "Code downloaded successfully",
      status: "success",
      duration: 2000,
      isClosable: true,
    });
  };

  // Get the code when the modal is opened
  useEffect(() => {
    if (isOpen && identifier) {
      fetchCode();
    }
  }, [isOpen, identifier]);

  // clear results
  const clearResults = () => {
    setExecutionResults([]);
  };

  // Generate modal title
  const modalTitle = title || `Code Editor - ${identifier}`;

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="full">
      <ModalOverlay />
      <ModalContent maxW="90vw" maxH="85vh" mx="auto" my="auto">
        <ModalHeader>
          <HStack justify="space-between" align="center">
            <VStack align="start" spacing={1}>
              <Text>{modalTitle}</Text>
              <HStack spacing={2}>
                <Badge colorScheme="blue">{language.toUpperCase()}</Badge>
                {hasUnsavedChanges && (
                  <Badge colorScheme="orange">Unsaved Changes</Badge>
                )}
                {!canEdit && (
                  <Badge colorScheme="gray">
                    {canSaveCopy ? 'Not your node: save it as your own copy' : 'Read only'}
                  </Badge>
                )}
                {source === 'library' && canEdit && (
                  <Badge colorScheme="purple">Library code: saving makes a copy for this project only</Badge>
                )}
                {source === 'project' && (
                  <Badge colorScheme="purple">Project copy: the library node is unchanged</Badge>
                )}
              </HStack>
            </VStack>
            
            <HStack spacing={2}>
              <Tooltip label="Copy Code">
                <IconButton
                  icon={<CopyIcon />}
                  size="sm"
                  variant="outline"
                  onClick={copyCode}
                  aria-label="Copy Code"
                />
              </Tooltip>
              
              <Tooltip label="Download Code">
                <IconButton
                  icon={<DownloadIcon />}
                  size="sm"
                  variant="outline"
                  onClick={downloadCode}
                  aria-label="Download Code"
                />
              </Tooltip>
              
              {showExecute && endpoints.executeCode && (
                <Button
                  leftIcon={<ArrowRightIcon />}
                  colorScheme="green"
                  size="sm"
                  onClick={executeCode}
                  isLoading={isExecuting}
                  loadingText="Executing"
                >
                  Run Code
                </Button>
              )}
              
              {source === 'project' && canEdit && endpoints.revertCode && (
                <Popover placement="bottom-end">
                  {({ onClose: closePopover }) => (
                    <>
                      <PopoverTrigger>
                        <Button size="sm" variant="outline" isDisabled={isSaving}>
                          Use library version
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent>
                        <PopoverArrow />
                        <PopoverBody>
                          <Text fontSize="sm" mb={2}>
                            Delete this project's copy? Its edits are lost and the
                            project goes back to the library code.
                          </Text>
                          <Button
                            size="sm"
                            colorScheme="red"
                            onClick={() => {
                              closePopover();
                              void revertToLibrary();
                            }}
                          >
                            Delete copy
                          </Button>
                        </PopoverBody>
                      </PopoverContent>
                    </>
                  )}
                </Popover>
              )}

              {canSaveCopy ? (
                <>
                  <Tooltip label="Class name of your copy (also its file name)">
                    <Input
                      size="sm"
                      width="16rem"
                      value={copyClassName}
                      onChange={(e) => setCopyClassName(e.target.value)}
                      aria-label="Class name of the copy"
                    />
                  </Tooltip>
                  <Button
                    colorScheme="blue"
                    size="sm"
                    onClick={saveAsCopy}
                    isLoading={isSaving}
                    loadingText="Saving"
                    isDisabled={!copyClassName.trim()}
                  >
                    Save as my copy
                  </Button>
                </>
              ) : (
                <Button
                  colorScheme="blue"
                  size="sm"
                  onClick={saveCode}
                  isLoading={isSaving}
                  loadingText="Saving"
                  isDisabled={!hasUnsavedChanges || !canEdit || isConflict}
                >
                  Save
                </Button>
              )}
            </HStack>
          </HStack>
        </ModalHeader>
        
        <ModalCloseButton />
        
        <ModalBody p={0} display="flex" flexDirection="column" height="calc(85vh - 120px)">
          {warnings.map((warning) => (
            <Alert status="info" key={warning}>
              <AlertIcon />
              <AlertDescription>{warning}</AlertDescription>
            </Alert>
          ))}
          {isConflict && (
            <Alert status="warning">
              <AlertIcon />
              <AlertDescription flex={1}>
                This file was changed elsewhere (for example in JupyterLab) after you opened it.
                Copy your edits if you need them, then reload.
              </AlertDescription>
              <Button size="sm" onClick={fetchCode}>
                Reload
              </Button>
            </Alert>
          )}
          <Tabs variant="enclosed" flex={1} display="flex" flexDirection="column">
            <TabList>
              <Tab>Code Editor</Tab>
              {showExecute && endpoints.executeCode && (
                <Tab>
                  Execution Results
                  {executionResults.length > 0 && (
                    <Badge ml={2} colorScheme="purple">
                      {executionResults.length}
                    </Badge>
                  )}
                </Tab>
              )}
            </TabList>
            
            <TabPanels flex={1} display="flex">
              <TabPanel flex={1} p={0}>
                {isLoading ? (
                  <Box
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    height="100%"
                  >
                    <VStack spacing={4}>
                      <Spinner size="lg" color="blue.500" />
                      <Text>Loading code...</Text>
                    </VStack>
                  </Box>
                ) : (
                  <Box
                    ref={containerRef}
                    height="100%"
                    border="1px solid"
                    borderColor="gray.200"
                    borderRadius="md"
                  />
                )}
              </TabPanel>
              
              {showExecute && endpoints.executeCode && (
                <TabPanel flex={1} p={4}>
                  <VStack spacing={4} align="stretch" height="100%">
                    <HStack justify="space-between">
                      <Text fontWeight="bold" fontSize="lg">
                        Execution Results
                      </Text>
                      {executionResults.length > 0 && (
                        <Button size="sm" variant="outline" onClick={clearResults}>
                          Clear Results
                        </Button>
                      )}
                    </HStack>
                    
                    <Box flex={1} overflowY="auto">
                      {executionResults.length === 0 ? (
                        <Alert status="info">
                          <AlertIcon />
                          <AlertTitle>No executions yet</AlertTitle>
                          <AlertDescription>
                            Click "Run Code" to execute your code and see results here.
                          </AlertDescription>
                        </Alert>
                      ) : (
                        <VStack spacing={4} align="stretch">
                          {executionResults.map((result, index) => (
                            <Box
                              key={index}
                              p={4}
                              border="1px solid"
                              borderColor={result.status === 'success' ? 'green.200' : 'red.200'}
                              borderRadius="md"
                              bg={result.status === 'success' ? 'green.50' : 'red.50'}
                            >
                              <HStack justify="space-between" mb={2}>
                                <Badge
                                  colorScheme={result.status === 'success' ? 'green' : 'red'}
                                >
                                  {result.status === 'success' ? 'Success' : 'Error'}
                                </Badge>
                                <Text fontSize="sm" color="gray.600">
                                  {result.timestamp} • {result.execution_time}ms
                                </Text>
                              </HStack>
                              
                              {result.output && (
                                <Box>
                                  <Text fontWeight="bold" fontSize="sm" mb={1}>
                                    Output:
                                  </Text>
                                  <Code
                                    display="block"
                                    p={2}
                                    bg="gray.100"
                                    borderRadius="md"
                                    whiteSpace="pre-wrap"
                                    fontSize="sm"
                                  >
                                    {result.output}
                                  </Code>
                                </Box>
                              )}
                              
                              {result.error && (
                                <Box>
                                  <Text fontWeight="bold" fontSize="sm" mb={1} color="red.600">
                                    Error:
                                  </Text>
                                  <Code
                                    display="block"
                                    p={2}
                                    bg="red.100"
                                    borderRadius="md"
                                    whiteSpace="pre-wrap"
                                    fontSize="sm"
                                    color="red.800"
                                  >
                                    {result.error}
                                  </Code>
                                </Box>
                              )}
                            </Box>
                          ))}
                        </VStack>
                      )}
                    </Box>
                  </VStack>
                </TabPanel>
              )}
            </TabPanels>
          </Tabs>
        </ModalBody>
        
        <ModalFooter>
          <HStack spacing={3}>
            <Text fontSize="sm" color="gray.600">
              {code.split('\n').length} lines
            </Text>
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </HStack>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

// Helper functions to dynamically load scripts
const loadScript = (src: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = () => resolve();
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

// global type definition
declare global {
  interface Window {
    monaco: any;
    require: any;
  }
}
