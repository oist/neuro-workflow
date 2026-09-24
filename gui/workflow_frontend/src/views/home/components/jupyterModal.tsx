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
  Text,
  Spinner,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  HStack,
  Badge,
  IconButton,
  Tooltip,
  VStack,
  useToast,
  Code,
  useColorModeValue,
} from '@chakra-ui/react';
import { ExternalLinkIcon, RepeatIcon, SettingsIcon, CopyIcon } from '@chakra-ui/icons';
import { JUPYTER_BASE_URL } from '../../../config/urls';
import { getJupyterSession, jupyterTreeUrl } from '../../../api/jupyterTenant';

interface JupyterModalProps {
  isOpen: boolean;
  onClose: () => void;
  projectId: string | null;
  title?: string;
  jupyterBaseUrl?: string;
  isDevelopment?: boolean; // Switch development mode
  jwtToken?: string; // JWT tokens for production environments
}

interface JupyterStatus {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  url: string | null;
}

const JupyterModal: React.FC<JupyterModalProps> = ({
  isOpen,
  onClose,
  projectId,
  title = "Jupyter Lab",
  jupyterBaseUrl = JUPYTER_BASE_URL,
  isDevelopment = true, // Default is development mode
  jwtToken, // For production environment
}) => {

  const toast = useToast();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [status, setStatus] = useState<JupyterStatus>({
    isLoading: false,
    isReady: false,
    error: null,
    url: null
  });

  const bg = useColorModeValue('white', 'gray.800');
  const headerBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('#e5e5e5', 'gray.200');
  const panelBg = useColorModeValue('#f7f7f8', 'gray.50');
  const subtextColor = useColorModeValue('gray.500', 'gray.600');
  const closeButtonBg = useColorModeValue('white', 'gray.700');
  const closeButtonHoverBg = useColorModeValue('gray.100', 'gray.600');

  // Launch JupyterHub and get the URL
  const initializeJupyter = async () => {
    if (!projectId) {
      setStatus({
        isLoading: false,
        isReady: false,
        error: "No project ID specified",
        url: null
      });
      return;
    }

    setStatus(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const session = await getJupyterSession();
      const tree = projectId
        ? `codes/projects/${projectId}/`
        : "codes/projects/";
      const jupyterUrl = jupyterTreeUrl(tree, session);

      console.log(`Initializing Jupyter for project ${projectId}`);
      console.log(`URL: ${jupyterUrl}`);

      setStatus({
        isLoading: false,
        isReady: true,
        error: null,
        url: jupyterUrl
      });

      toast({
        title: "Jupyter Lab Ready",
        description: isDevelopment
          ? `Project "${projectId}" JupyterLab has started (development mode)`
          : `Project "${projectId}" JupyterLab has started`,
        status: "success",
        duration: 3000,
        isClosable: true,
      });

    } catch (error) {
      console.error('JupyterHub initialization error:', error);

      const errorMessage = error instanceof Error ? error.message : "Startup failed";

      setStatus({
        isLoading: false,
        isReady: false,
        error: errorMessage,
        url: null
      });

      toast({
        title: "JupyterHub startup error",
        description: errorMessage,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Wait for JupyterHub to be ready
  const waitForJupyterReady = async (
    baseUrl: string,
    projectId: string,
    maxAttempts = 30
  ): Promise<void> => {
    console.log(`Waiting for JupyterHub to be ready for project ${projectId}...`);

    for (let i = 0; i < maxAttempts; i++) {
      try {
        // Use a health check endpoint to avoid CORS errors
        const healthCheckUrl = `${baseUrl}/hub/api`;

        await fetch(healthCheckUrl, {
          method: 'HEAD',
          mode: 'no-cors' // Avoid CORS errors
        });

        // In no-cors mode, an opaque response is always returned.
        // Actual startup check is performed on a time basis
        if (i >= 3) { // Wait at least 3 seconds
          console.log(`JupyterHub is ready for project ${projectId}`);
          return;
        }
      } catch (error) {
        // Ignore the error and continue
      }

      // Wait 1 second
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    throw new Error('JupyterHub startup timed out');
  };

  // Initialize Jupyter when modal is opened
  useEffect(() => {
    if (isOpen && projectId && !status.isReady && !status.isLoading) {
      initializeJupyter();
    }
  }, [isOpen, projectId]);

  // Reset status if project ID changes
//   useEffect(() => {
//     if (!isOpen || !projectId) {
//       setStatus({
//         isLoading: false,
//         isReady: false,
//         error: null,
//         url: null
//       });
//     }
//   }, [isOpen, projectId]);

  // retry
  const handleRetry = () => {
    initializeJupyter();
  };

  // Open in new tab
  const handleOpenInNewTab = () => {
    if (status.url) {
      window.open(status.url, '_blank');
    }
  };

  // Copy URL
  const handleCopyUrl = () => {
    if (status.url) {
      navigator.clipboard.writeText(status.url);
      toast({
        title: "copy",
        status: "success",
        duration: 2000,
        isClosable: true,
      });
    }
  };

  // iframe Load error handler for
  const handleIframeError = () => {
    console.error('iframe load error');
    setStatus(prev => ({
      ...prev,
      error: "JupyterLab failed to load"
    }));
  };

  // iframe Load success handler for
  const handleIframeLoad = () => {
    console.log('iframe loaded successfully');
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="full"
      closeOnOverlayClick={false}
    >
      <ModalOverlay bg="blackAlpha.600" />
      <ModalContent
        maxW="95vw"
        maxH="95vh"
        m={4}
        bg={bg}
        borderRadius="lg"
        overflow="hidden"
      >
        <ModalHeader
          bg={headerBg}
          borderBottom="1px"
          borderColor={borderColor}
          py={3}
        >
          <HStack justify="space-between" align="center">
            <HStack spacing={3}>
              <Text fontWeight="bold" fontSize="sm">
                {title}
              </Text>
              {projectId && (
                <Badge colorScheme="purple" variant="subtle">
                  Project: {projectId}
                </Badge>
              )}

              {/* Development/Production Mode Display */}
              <Badge
                colorScheme={isDevelopment ? "green" : "blue"}
                variant="outline"
                size="sm"
              >
                {isDevelopment ? "Development" : "Production"}
              </Badge>
            </HStack>

            <HStack spacing={2}>
              {status.isReady && (
                <>
                  <Tooltip label="Copy URL">
                    <IconButton
                      aria-label="Copy URL"
                      icon={<CopyIcon />}
                      size="sm"
                      variant="ghost"
                      onClick={handleCopyUrl}
                    />
                  </Tooltip>

                  <Tooltip label="Open in new tab">
                    <IconButton
                      aria-label="Open in new tab"
                      icon={<ExternalLinkIcon />}
                      size="sm"
                      variant="ghost"
                      onClick={handleOpenInNewTab}
                    />
                  </Tooltip>

                  <Tooltip label="reload">
                    <IconButton
                      aria-label="reload"
                      icon={<RepeatIcon />}
                      size="sm"
                      variant="ghost"
                      onClick={handleRetry}
                    />
                  </Tooltip>
                </>
              )}

              <Tooltip label="setting">
                <IconButton
                  aria-label="setting"
                  icon={<SettingsIcon />}
                  size="sm"
                  variant="ghost"
                  isDisabled
                />
              </Tooltip>
            </HStack>
          </HStack>
        </ModalHeader>

        <ModalCloseButton
          size="lg"
          top={2}
          right={2}
          bg={closeButtonBg}
          _hover={{ bg: closeButtonHoverBg }}
        />

        <ModalBody p={0} bg={panelBg}>
          {status.isLoading && (
            <VStack
              justify="center"
              align="center"
              h="70vh"
              spacing={4}
            >
              <Spinner size="xl" color="purple.500" thickness="4px" />
              <VStack spacing={2} textAlign="center">
                <Text fontSize="lg" fontWeight="semibold">
                  Starting JupyterLab...
                </Text>
                <Text fontSize="sm" color={subtextColor}>
                  Project ID: <Code>{projectId}</Code>
                </Text>
                <Text fontSize="sm" color={subtextColor}>
                  {isDevelopment
                    ? "Development mode (no authentication, automatic login)"
                    : "Production mode (JWT authentication and user environment preparation in progress)"
                  }
                </Text>
              </VStack>
            </VStack>
          )}

          {status.error && (
            <Box p={8}>
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                <Box>
                  <AlertTitle>startup error</AlertTitle>
                  <AlertDescription mt={2}>
                    {status.error}
                  </AlertDescription>
                  {projectId && (
                    <Text fontSize="sm" color={subtextColor} mt={2}>
                      Project ID: <Code>{projectId}</Code>
                    </Text>
                  )}
                </Box>
              </Alert>

              <HStack mt={4} justify="center">
                <Button
                  colorScheme="red"
                  variant="outline"
                  onClick={handleRetry}
                  leftIcon={<RepeatIcon />}
                >
                  Retry
                </Button>
                <Button variant="ghost" onClick={onClose}>
                  Close
                </Button>
              </HStack>
            </Box>
          )}

          {status.isReady && status.url && (
            <Box h="calc(95vh - 120px)" w="100%">
              <iframe
                ref={iframeRef}
                src={status.url}
                width="100%"
                height="100%"
                style={{
                  border: 'none',
                  borderRadius: '0 0 8px 8px',
                  backgroundColor: 'white'
                }}
                title={`Jupyter Lab - Project ${projectId}`}
                onError={handleIframeError}
                onLoad={handleIframeLoad}
                sandbox="allow-same-origin allow-scripts allow-forms allow-downloads allow-modals allow-popups allow-popups-to-escape-sandbox"
              />
            </Box>
          )}
        </ModalBody>

        <ModalFooter
          bg={panelBg}
          borderTop="1px"
          borderColor={borderColor}
          py={2}
        >
          <HStack spacing={3} justify="space-between" w="100%">
            <HStack spacing={2}>
              {status.isReady && (
                <>
                  <Text fontSize="xs" color={subtextColor}>
                    💡 Tip: Ctrl+S Save the notebook with
                  </Text>
                  <Text fontSize="xs" color={subtextColor}>|</Text>
                  <Text fontSize="xs" color={subtextColor}>
                    📁 working folder: /projects/{projectId}
                  </Text>
                </>
              )}
            </HStack>

            <Button variant="ghost" onClick={onClose} size="sm">
              閉じる
            </Button>
          </HStack>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default JupyterModal;
