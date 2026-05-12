import {
  HStack,
  Box,
  Button,
  Text,
  VStack,
  Badge,
  IconButton,
} from '@chakra-ui/react';
import { ViewIcon } from '@chakra-ui/icons';
import { FiMenu } from 'react-icons/fi';
import { Project } from './type';
import { useFlowStore, FlowStore } from '../../stores/flowStore';

interface WorkflowToolbarProps {
  isIslandCodeOpen: boolean;
  setIslandCodeOpen: (open: boolean) => void;
  isConnected: boolean;
  autoSaveEnabled: boolean;
  selectedProject: string | null;
  projects: Project[];
  isGeneratingCode: boolean;
  handleOpenJupyter: () => void;
  handleGenerateCode: () => void;
  handleExportFlowJSON: () => void;
}

export const WorkflowToolbar: React.FC<WorkflowToolbarProps> = ({
  isIslandCodeOpen,
  setIslandCodeOpen,
  isConnected,
  autoSaveEnabled,
  selectedProject,
  projects,
  isGeneratingCode,
  handleOpenJupyter,
  handleGenerateCode,
  handleExportFlowJSON,
}) => {
  // Subscribe only to the node count, not the full sharedNodes array, so the
  // toolbar does not re-render on every drag frame. Length changes only when
  // nodes are added or removed.
  const nodesCount = useFlowStore((state: FlowStore) => state.sharedNodes.length);
  return (
    <>
      <IconButton
        position="absolute"
        top="16px"
        right="16px"
        zIndex={1000}
        aria-label="Open/close menu"
        icon={<FiMenu />}
        onClick={() => setIslandCodeOpen(!isIslandCodeOpen)}
        colorScheme="gray"
        bg="gray.300"
        _hover={{ bg: 'gray.600' }}
      />
      {/* explanation */}
      <Box
        position="absolute"
        top="10px"
        right="10px"
        display={isIslandCodeOpen ? 'block' : 'none'}
        p={4}
        bg="white"
        borderRadius="lg"
        boxShadow="lg"
        maxWidth="340px"
        zIndex={5}
        borderWidth={1}
        borderColor="gray.200"
      >
        <VStack spacing={4} align="stretch">
          {/* header */}
          <Box paddingRight={12}>
            <HStack justify="space-between" align="center">
              <Text fontWeight="bold" fontSize="md" color="gray.800">
                🔬 Flow Designer
              </Text>
              {isConnected ? (
                <Badge colorScheme="green" size="sm" variant="subtle">
                  Online
                </Badge>
              ) : (
                <Badge colorScheme="red" size="sm" variant="subtle">
                  Offline
                </Badge>
              )}
            </HStack>
          </Box>
          {/* Explanatory text */}
          <Box>
            <Text fontSize="sm" color="gray.600" lineHeight="1.4">
              Drag nodes from the left panel to build mathematical workflows. Connect outputs to inputs to create calculations.
            </Text>
          </Box>

          {/* Tips & Status */}
          <Box>
            <Text fontSize="xs" color="blue.600" mb={1}>
              💡 Tips: Click edges to delete • Press Delete key for selected items
            </Text>
            {!autoSaveEnabled && (
              <Text fontSize="xs" color="orange.600">
                ⚠️ Auto-save disabled
              </Text>
            )}
          </Box>

          {/* Action Buttons */}
          <VStack spacing={2} align="stretch">
            <Button
              leftIcon={<ViewIcon />}
              colorScheme="purple"
              variant="outline"
              size="sm"
              onClick={handleOpenJupyter}
              isDisabled={!selectedProject}
              _hover={{ bg: "purple.50", transform: "translateY(-1px)" }}
              _disabled={{
                opacity: 0.4,
                cursor: "not-allowed"
              }}
              transition="all 0.2s"
            >
              {selectedProject ? "🚀 Open JupyterLab Tab" : "Select Project First"}
            </Button>

            <Button
              colorScheme="blue"
              variant="solid"
              size="sm"
              onClick={handleGenerateCode}
              isDisabled={!selectedProject || nodesCount === 0}
              isLoading={isGeneratingCode}
              loadingText="Generating..."
              _hover={{ bg: "blue.600", transform: "translateY(-1px)" }}
              _disabled={{
                opacity: 0.4,
                cursor: "not-allowed"
              }}
              transition="all 0.2s"
            >
              {!selectedProject ? "Select Project First" :
              nodesCount === 0 ? "Add Nodes to Generate" :
              "📝 Generate Code"}
            </Button>

            <Button
              colorScheme="green"
              variant="outline"
              size="sm"
              onClick={handleExportFlowJSON}
              isDisabled={!selectedProject || nodesCount === 0}
              _hover={{ bg: "green.50", transform: "translateY(-1px)" }}
              _disabled={{
                opacity: 0.4,
                cursor: "not-allowed"
              }}
              transition="all 0.2s"
            >
              {nodesCount === 0 ? "No Flow to Export" : "📋 Export Flow JSON"}
            </Button>

            {selectedProject && (
              <Text fontSize="xs" color="gray.500" textAlign="center">
                Project: {projects.find(p => p.id === selectedProject)?.name || 'Unknown'}
              </Text>
            )}
          </VStack>
        </VStack>
      </Box>
    </>
  );
};

export default WorkflowToolbar;
