import { useEffect, useState, useCallback, useRef } from 'react';
import { HpcTarget, Project, Visibility } from '../type'
import {
  HStack,
  Box,
  VStack,
  Select,
  FormLabel,
  Badge,
  Text,
  Flex,
  IconButton,
  Tooltip,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Textarea,
  Button,
  RadioGroup,
  Radio,
  Stack,
  useDisclosure,
} from '@chakra-ui/react';
import { CheckIcon, WarningIcon, DeleteIcon, EditIcon } from '@chakra-ui/icons';
import { FiMenu, FiPlay } from 'react-icons/fi';
import { useTabContext } from '../../../components/tabs/TabManager';
import { useAuth } from '../../../auth/authContext';
import { createAuthHeaders } from '../../../api/authHeaders';
import LogViewModal, { LogEntry } from "./logViewModal";
import { runWorkflowStream } from '../../../api/workflowRunApi';
import { WorkflowContextEditor } from '../../../components/WorkflowContextEditor';
import { JUPYTER_BASE_URL } from '../../../config/urls';

export const ProjectSelector = ({ 
  projects, 
  selectedProject, 
  onProjectChange, 
  onProjectDelete,
  onProjectUpdate,
  autoSaveEnabled = true,
  isConnected = true 
}: {
  projects: Project[];
  selectedProject: string | null;
  onProjectChange: (projectId: string) => void;
  onProjectDelete?: (project: Project) => void;
  onProjectUpdate?: (projectId: string, updates: Partial<Project>) => void;
  autoSaveEnabled?: boolean;
  isConnected?: boolean;
}) => {
  const toast = useToast();
  const { isOpen: isContextOpen, onOpen: onContextOpen, onClose: onContextClose } = useDisclosure();
  const [contextInitial, setContextInitial] = useState<Record<string, any>>({});
  const [contextDraft, setContextDraft] = useState<Record<string, any> | null>(null);
  const [isContextValid, setIsContextValid] = useState<boolean>(true);
  const [contextResetKey, setContextResetKey] = useState<number>(0);
  const [descriptionDraft, setDescriptionDraft] = useState<string>('');
  const [visibilityDraft, setVisibilityDraft] = useState<Visibility>('private');
  const [referenceDraft, setReferenceDraft] = useState<string>('');
  const [hpcTargetDraft, setHpcTargetDraft] = useState<HpcTarget>('');
  const currentProject = selectedProject
    ? projects.find((p) => p.id === selectedProject) ?? null
    : null;
  // Island menu opening/closing management
  const [isIslandProjectOpen, setIslandProjectOpen] = useState(true);
  // Use the tab system context
  const { addJupyterTab } = useTabContext();
  // Log View Modal - streaming state
  const [isLogOpen, setIsLogOpen] = useState<boolean>(false);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [runStatus, setRunStatus] = useState<"idle" | "running" | "ok" | "error">("idle");
  const abortControllerRef = useRef<AbortController | null>(null);
  const { registerRunController } = useAuth();

  useEffect(() => {
    registerRunController({
      isRunning,
      abort: () => {
        abortControllerRef.current?.abort();
        abortControllerRef.current = null;
        setIsRunning(false);
        setRunStatus("idle");
      },
    });
    return () => registerRunController(null);
  }, [isRunning, registerRunController]);

  // Helper functions for API communication
  const createAuthHeadersLocal = async () => {
    return await createAuthHeaders();
  };

  const handleSaveContext = async () => {
    if (!selectedProject) {
      toast({
        title: "No Project Selected",
        description: "Please select a project first",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    if (!isContextValid) {
      toast({
        title: "Invalid JSON",
        description: "Workflow context must be valid JSON.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    const parsedContext = contextDraft ?? {};
    const descriptionPayload = descriptionDraft.trim();
    const payload: Record<string, any> = {
      workflow_context: parsedContext,
      description: descriptionPayload,
      reference: referenceDraft,
      hpc_target: hpcTargetDraft,
    };
    if (currentProject?.can_change_visibility && visibilityDraft !== currentProject.visibility) {
      payload.visibility = visibilityDraft;
    }

    try {
      const headers = await createAuthHeadersLocal();
      const response = await fetch(`/api/workflow/${selectedProject}/`, {
        method: "PATCH",
        credentials: "include",
        headers: {
          ...headers,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData?.error || "Failed to update workflow context");
      }

      const updated: Partial<Project> = await response.json();
      onProjectUpdate?.(selectedProject, updated);
      toast({
        title: "Context Updated",
        description: "Workflow context saved successfully.",
        status: "success",
        duration: 3000,
        isClosable: true,
      });
      setContextInitial(parsedContext);
      setContextDraft(parsedContext);
      if (updated.visibility) {
        setVisibilityDraft(updated.visibility);
      }
      if (updated.reference !== undefined) {
        setReferenceDraft(updated.reference);
      }
      if (updated.hpc_target !== undefined) {
        setHpcTargetDraft(updated.hpc_target);
      }
      onContextClose();
    } catch (error) {
      toast({
        title: "Update Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };

  // Get status badge
  const getStatusBadge = () => {
    if (!isConnected) {
      return (
        <Badge colorScheme="red" size="sm" variant="subtle">
          <HStack spacing={1}>
            <WarningIcon w={2} h={2} />
            <Text fontSize="xs">Offline</Text>
          </HStack>
        </Badge>
      );
    }
    
    if (autoSaveEnabled) {
      return (
        <Badge colorScheme="green" size="sm" variant="subtle">
          <HStack spacing={1}>
            <CheckIcon w={2} h={2} />
            <Text fontSize="xs">Auto-save</Text>
          </HStack>
        </Badge>
      );
    }
    
    return (
      <Badge colorScheme="orange" size="sm" variant="subtle">
        <HStack spacing={1}>
          <WarningIcon w={2} h={2} />
          <Text fontSize="xs">Manual save</Text>
        </HStack>
      </Badge>
    );
  };

  // Get status message
  const getStatusMessage = () => {
    if (!isConnected) {
      return "⚠️ Connection lost - changes not saved";
    }
    if (autoSaveEnabled) {
      return "💾 Changes are automatically saved";
    }
    return "⚠️ Remember to save manually";
  };

  // Viewing the source code of a workflow project
  const handleOpenJupyter = useCallback(async () => {
    if (!selectedProject) {
      toast({
        title: "No Project Selected",
        description: "Please select a project first",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    try {
      let projectId = localStorage.getItem('projectId');
      projectId = projectId ? projectId : "";
      // Get project name
      const projectName = projects.find(p => p.id === selectedProject)?.name || selectedProject;
      // Initial capitalization
      const trimedProjectName = projectName.replace(/\s/g, '').toLowerCase();
      const capitalizedProjectName = trimedProjectName.charAt(0).toUpperCase() + trimedProjectName.slice(1);

      const jupyterUrl = `${JUPYTER_BASE_URL}/user/user1/lab/workspaces/auto-E/tree/codes/projects/${capitalizedProjectName}/${capitalizedProjectName}.py`;
      
      // Create new tab
      addJupyterTab(selectedProject, projectName, jupyterUrl);
      
      toast({
        title: "JupyterLab Tab Created",
        description: `Created tab for project "${projectName}"`,
        status: "success",
        duration: 2000,
        isClosable: true,
      });
      
    } catch (error) {
      console.error('Error creating JupyterLab tab:', error);
      toast({
        title: "Error",
        description: "Failed to create JupyterLab tab",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  }, [selectedProject, projects, addJupyterTab, toast]);

  // Stop a running workflow
  const handleStopWorkflow = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsRunning(false);
    setRunStatus("idle");
    setLogEntries(prev => [...prev, { type: "info", content: "\n--- Execution cancelled by user ---\n" }]);
  }, []);

  // Track whether a "done" event was received (avoids stale closure on runStatus)
  const receivedDoneRef = useRef(false);

  // Run workflow project (SSE streaming)
  const handleRunWorkflow = useCallback(async () => {
    if (!selectedProject) {
      toast({
        title: "No Project Selected",
        description: "Please select a project first",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    // Prevent concurrent executions
    if (isRunning) {
      toast({
        title: "Already Running",
        description: "A workflow is already running. Stop it first.",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    // Reset state and open modal
    setLogEntries([]);
    setIsRunning(true);
    setRunStatus("running");
    setIsLogOpen(true);
    receivedDoneRef.current = false;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await runWorkflowStream(
        selectedProject,
        (event) => {
          switch (event.type) {
            case "run_started":
              setLogEntries(prev => [
                ...prev,
                { type: "info", content: `Starting workflow: ${event.data.project_name}\n` },
              ]);
              break;
            case "stdout":
              setLogEntries(prev => [
                ...prev,
                { type: "stdout", content: event.data.content as string },
              ]);
              break;
            case "stderr":
              setLogEntries(prev => [
                ...prev,
                { type: "stderr", content: event.data.content as string },
              ]);
              break;
            case "execute_result":
              setLogEntries(prev => [
                ...prev,
                { type: "execute_result", content: event.data.content as string },
              ]);
              break;
            case "error": {
              const tb = (event.data.traceback as string[]) || [];
              const errorText = tb.length > 0
                ? tb.join("\n")
                : `${event.data.ename}: ${event.data.evalue}`;
              setLogEntries(prev => [
                ...prev,
                { type: "error", content: errorText + "\n" },
              ]);
              break;
            }
            case "done": {
              receivedDoneRef.current = true;
              const status = event.data.status as string;
              setRunStatus(status === "ok" ? "ok" : "error");
              setIsRunning(false);
              setLogEntries(prev => [
                ...prev,
                { type: "info", content: `\n--- Execution finished (${status}) ---\n` },
              ]);
              break;
            }
          }
        },
        controller.signal
      );

      // Stream ended – only update if no "done" event was received
      if (!receivedDoneRef.current) {
        setIsRunning(false);
        setRunStatus("ok");
      }
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        // User cancelled – already handled in handleStopWorkflow
        return;
      }
      console.error("Error running workflow:", error);
      setIsRunning(false);
      setRunStatus("error");
      setLogEntries(prev => [
        ...prev,
        { type: "error", content: `\nConnection error: ${(error as Error).message}\n` },
      ]);
      toast({
        title: "Error",
        description: (error as Error).message || "Failed to run workflow",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      abortControllerRef.current = null;
    }
  }, [selectedProject, toast, isRunning]);

  return (
    <Box
      position="absolute"
      top="8px"
      left="16px"
    >
      <HStack spacing={2}>
        <IconButton
          position="absolute"
          top="16px"
          left="16px"
          zIndex={1000}
          aria-label="Open/close menu"
          icon={<FiMenu />}
          onClick={() => setIslandProjectOpen(!isIslandProjectOpen)}
          colorScheme="gray"
          bg="gray.300"
          _hover={{ bg: 'gray.600' }}
        />
        <IconButton
          position="absolute"
          top="16px"
          left="64px"
          zIndex={1000}
          aria-label="Execute"
          icon={<FiPlay />}
          onClick={() => handleRunWorkflow()}
          colorScheme="gray"
          bg="pink.300"
          _hover={{ bg: 'gray.600' }}
        />
      </HStack>
      <Box
        position="absolute"
        top="0px"
        left="0px"
        p={4}
        bg="white"
        borderRadius="md"
        boxShadow="md"
        zIndex={5}
        minWidth="320px"
        borderWidth={1}
        borderColor="gray.200"
        display={isIslandProjectOpen ? 'block' : 'none'}
      >
        <VStack spacing={3} align="stretch" width="100%">
          {/* header part */}
          <Flex justify="space-between" align="center">
            <Box marginLeft={24}>
              <FormLabel fontSize="sm" mb={0} color="gray.700" fontWeight="semibold">
                Project Workspace
              </FormLabel>
              <HStack spacing={2}>
                {getStatusBadge()}
                {currentProject && (
                  <Badge
                    colorScheme={currentProject.visibility === 'public' ? 'teal' : 'gray'}
                    size="sm"
                    variant="subtle"
                  >
                    {currentProject.visibility === 'public' ? 'Public' : 'Private'}
                  </Badge>
                )}
              </HStack>
            </Box>
          </Flex>
          
          {/* Project selection */}
          <HStack spacing={2}>
            <Select 
              value={selectedProject || ''} 
              onChange={(e) => onProjectChange(e.target.value)}
              onLoad={(e) => onProjectChange(e.target.value)}
              size="sm"
              bg="white"
              color="gray.800"
              borderColor="gray.300"
              placeholder="Choose a project..."
              _hover={{
                borderColor: "blue.300"
              }}
              _focus={{
                borderColor: "blue.500",
                boxShadow: "0 0 0 1px #3182ce"
              }}
              flex="1"
              marginTop={2}
            >
              {projects.map(project => {
                const visLabel =
                  project.visibility === 'public' ? ' (Public)' : '';
                return (
                  <option key={project.id} value={project.id} style={{ color: '#2D3748' }}>
                    {project.name}
                    {visLabel}
                  </option>
                );
              })}
            </Select>

            {selectedProject && (
              <Tooltip
                label={(() => {
                  const project = projects.find(p => p.id === selectedProject);
                  const ctx = project?.workflow_context ?? {};
                  const species = ctx.species ? `species: ${ctx.species}` : "species: -";
                  const sources = Array.isArray(ctx.metadata_sources) && ctx.metadata_sources.length > 0
                    ? `sources: ${ctx.metadata_sources.join(", ")}`
                    : "sources: -";
                  const resources = ctx.resource_requirements || {};
                  const cpus = resources.cpus !== undefined ? `cpus: ${resources.cpus}` : "cpus: -";
                  const mem = resources.memory_gb !== undefined ? `mem_gb: ${resources.memory_gb}` : "mem_gb: -";
                  return `${species} | ${sources} | ${cpus} | ${mem}`;
                })()}
                placement="top"
              >
                <IconButton
                  aria-label="Edit workflow context"
                  icon={<EditIcon />}
                  size="sm"
                  colorScheme="blue"
                  variant="outline"
                  onClick={() => {
                    const project = projects.find(p => p.id === selectedProject);
                    const context = project?.workflow_context ?? {};
                    setDescriptionDraft(project?.description ?? '');
                    setContextInitial(context);
                    setContextDraft(context);
                    setIsContextValid(true);
                    setContextResetKey(prev => prev + 1);
                    setVisibilityDraft(project?.visibility ?? 'private');
                    setReferenceDraft(project?.reference ?? '');
                    setHpcTargetDraft(project?.hpc_target ?? '');
                    onContextOpen();
                  }}
                  _hover={{
                    bg: "blue.50",
                    borderColor: "blue.400"
                  }}
                  marginTop={2}
                />
              </Tooltip>
            )}
            
            {selectedProject && onProjectDelete && currentProject?.can_delete && (
              <Tooltip label="Delete project" placement="top">
                <IconButton
                  aria-label="Delete project"
                  icon={<DeleteIcon />}
                  size="sm"
                  colorScheme="red"
                  variant="outline"
                  onClick={() => {
                    const project = projects.find(p => p.id === selectedProject);
                    if (project) {
                      onProjectDelete(project);
                    }
                  }}
                  _hover={{
                    bg: "red.50",
                    borderColor: "red.400"
                  }}
                  marginTop={2}
                />
              </Tooltip>
            )}
          </HStack>
          
          {/* status message */}
          {selectedProject && (
            <Text 
              fontSize="xs" 
              color={!isConnected ? "red.600" : autoSaveEnabled ? "green.600" : "orange.600"}
              textAlign="center"
              py={1}
              px={2}
              bg={!isConnected ? "red.50" : autoSaveEnabled ? "green.50" : "orange.50"}
              borderRadius="sm"
              borderWidth={1}
              borderColor={!isConnected ? "red.200" : autoSaveEnabled ? "green.200" : "orange.200"}
            >
              {getStatusMessage()}
            </Text>
          )}LogViewDialog
          
          {/* Project information (displayed only when selected) */}
          {selectedProject && (
            <Box pt={2} borderTop="1px" borderColor="gray.100">
              <Text fontSize="xs" color="gray.500">
                Project ID: {selectedProject}
              </Text>
            </Box>
          )}
        </VStack>
      </Box>

      <LogViewModal
        isOpen={isLogOpen}
        onClose={() => {
          setIsLogOpen(false);
        }}
        logEntries={logEntries}
        isRunning={isRunning}
        runStatus={runStatus}
        onStop={handleStopWorkflow}
      />

      <Modal isOpen={isContextOpen} onClose={onContextClose} size="lg">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Workflow Context</ModalHeader>
          <ModalBody>
            <Box mb={4}>
              <FormLabel fontSize="sm">Project Description</FormLabel>
              <Textarea
                value={descriptionDraft}
                onChange={(e) => setDescriptionDraft(e.target.value)}
                placeholder="Describe your workflow project..."
              />
            </Box>
            <Box mb={4}>
              <FormLabel fontSize="sm">Visibility</FormLabel>
              <RadioGroup
                value={visibilityDraft}
                onChange={(value) => setVisibilityDraft(value as Visibility)}
                isDisabled={!currentProject?.can_change_visibility}
              >
                <Stack direction="row" spacing={6}>
                  <Radio value="private">Private</Radio>
                  <Radio value="public">Public</Radio>
                </Stack>
              </RadioGroup>
              {!currentProject?.can_change_visibility && (
                <Text fontSize="xs" color="gray.500" mt={1}>
                  Only the project owner can change visibility.
                </Text>
              )}
            </Box>
            <Box mb={4}>
              <FormLabel fontSize="sm">Reference</FormLabel>
              <Textarea
                value={referenceDraft}
                onChange={(e) => setReferenceDraft(e.target.value)}
                placeholder="Papers, URLs, or other references for this workflow..."
              />
            </Box>
            <Box mb={4}>
              <FormLabel fontSize="sm">HPC Target</FormLabel>
              <Select
                value={hpcTargetDraft}
                onChange={(e) => setHpcTargetDraft(e.target.value as HpcTarget)}
              >
                <option value="">Not specified</option>
                <option value="riken">Riken</option>
                <option value="fugaku">Fugaku</option>
              </Select>
            </Box>
            <WorkflowContextEditor
              key={contextResetKey}
              initialContext={contextInitial}
              onChange={(context, rawText, isValid) => {
                setContextDraft(context);
                setIsContextValid(isValid);
              }}
            />
          </ModalBody>
          <ModalFooter>
            <Button variant="ghost" mr={3} onClick={onContextClose}>
              Cancel
            </Button>
            <Button colorScheme="blue" onClick={handleSaveContext}>
              Save
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
};
