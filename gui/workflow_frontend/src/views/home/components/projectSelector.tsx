import { Project } from '../type'
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
} from '@chakra-ui/react';
import { CheckIcon, WarningIcon, DeleteIcon } from '@chakra-ui/icons';

export const ProjectSelector = ({ 
  projects, 
  selectedProject, 
  onProjectChange, 
  onProjectDelete,
  autoSaveEnabled = true,
  isConnected = true 
}: {
  projects: Project[];
  selectedProject: string | null;
  onProjectChange: (projectId: string) => void;
  onProjectDelete?: (project: Project) => void;
  autoSaveEnabled?: boolean;
  isConnected?: boolean;
}) => {

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

  const getStatusMessage = () => {
    if (!isConnected) {
      return "⚠️ Connection lost - changes not saved";
    }
    if (autoSaveEnabled) {
      return "💾 Changes are automatically saved";
    }
    return "⚠️ Remember to save manually";
  };

  return (
    <Box
      position="absolute"
      top="10px"
      left="10px"
      p={4}
      bg="white"
      borderRadius="md"
      boxShadow="md"
      zIndex={5}
      minWidth="320px"
      borderWidth={1}
      borderColor="gray.200"
    >
      <VStack spacing={3} align="stretch" width="100%">
        {/* ヘッダー部分 */}
        <Flex justify="space-between" align="center">
          <FormLabel fontSize="sm" mb={0} color="gray.700" fontWeight="semibold">
            Project Workspace
          </FormLabel>
          {getStatusBadge()}
        </Flex>
        
        {/* プロジェクト選択 */}
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
          >
            {projects.map(project => (
              <option key={project.id} value={project.id} style={{color: '#2D3748'}}>
                {project.name}
              </option>
            ))}
          </Select>
          
          {selectedProject && onProjectDelete && (
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
              />
            </Tooltip>
          )}
        </HStack>
        
        {/* ステータスメッセージ */}
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
        )}
        
        {/* プロジェクト情報（選択時のみ表示） */}
        {selectedProject && (
          <Box pt={2} borderTop="1px" borderColor="gray.100">
            <Text fontSize="xs" color="gray.500">
              Project ID: {selectedProject}
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
};
