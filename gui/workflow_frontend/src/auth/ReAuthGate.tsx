import { useCallback, useState } from 'react';
import {
  Alert,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  Button,
  Modal,
  ModalBody,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  UnorderedList,
  ListItem,
  VStack,
} from '@chakra-ui/react';
import { useAuth } from './authContext';
import { getKeycloak } from './keycloak';

const ReAuthGate = () => {
  const {
    reAuthRequired,
    reAuthReason,
    hasPendingSaves,
    runController,
    flushPendingSaves,
  } = useAuth();

  const [isWorking, setIsWorking] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const isRunning = !!runController?.isRunning;

  const goToLogin = useCallback(() => {
    // Bypass our wrapped signIn (which toggles loading state). Direct kc.login()
    // navigates the browser to Keycloak; React state after this is irrelevant.
    getKeycloak().login();
  }, []);

  const handlePrimary = useCallback(async () => {
    setErrorMsg(null);
    setIsWorking(true);
    try {
      if (isRunning) {
        runController?.abort();
      }
      if (hasPendingSaves) {
        await flushPendingSaves();
      }
      goToLogin();
    } catch (err) {
      console.error('Re-auth flush failed:', err);
      setErrorMsg(
        err instanceof Error
          ? `Failed to save before re-login: ${err.message}`
          : 'Failed to save before re-login.'
      );
      setIsWorking(false);
    }
  }, [isRunning, runController, hasPendingSaves, flushPendingSaves, goToLogin]);

  const handleSecondary = useCallback(() => {
    // Discard pending work and proceed to login immediately.
    goToLogin();
  }, [goToLogin]);

  let primaryLabel = 'Re-login';
  if (isRunning) {
    primaryLabel = 'Abort & re-login';
  } else if (hasPendingSaves) {
    primaryLabel = 'Save & re-login';
  }

  const showSecondary = isRunning || hasPendingSaves;

  const reasonText =
    reAuthReason === 'api-401'
      ? 'The server rejected the request (401). Your session may have expired.'
      : 'Your session has expired.';

  return (
    <Modal
      isOpen={reAuthRequired}
      onClose={() => {
        /* prevent close via overlay/Esc; user must pick a button */
      }}
      closeOnEsc={false}
      closeOnOverlayClick={false}
      isCentered
      size="md"
    >
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Session expired</ModalHeader>
        <ModalBody>
          <VStack align="stretch" spacing={3}>
            <Text>{reasonText}</Text>
            {(isRunning || hasPendingSaves) && (
              <Alert status="warning" variant="left-accent" borderRadius="md">
                <AlertIcon />
                <VStack align="start" spacing={1}>
                  <AlertTitle fontSize="sm">Save before signing back in</AlertTitle>
                  <AlertDescription fontSize="sm">
                    <UnorderedList spacing={1} pl={3}>
                      {isRunning && (
                        <ListItem>
                          A workflow run is in progress and will be aborted if you re-login now.
                        </ListItem>
                      )}
                      {hasPendingSaves && (
                        <ListItem>
                          Unsaved changes on the canvas may be lost.
                        </ListItem>
                      )}
                    </UnorderedList>
                  </AlertDescription>
                </VStack>
              </Alert>
            )}
            {errorMsg && (
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                <AlertDescription fontSize="sm">{errorMsg}</AlertDescription>
              </Alert>
            )}
          </VStack>
        </ModalBody>
        <ModalFooter gap={2}>
          {showSecondary && (
            <Button
              variant="ghost"
              onClick={handleSecondary}
              isDisabled={isWorking}
            >
              Sign in now (discard)
            </Button>
          )}
          <Button
            colorScheme="blue"
            onClick={handlePrimary}
            isLoading={isWorking}
            loadingText={hasPendingSaves ? 'Saving...' : 'Working...'}
          >
            {primaryLabel}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ReAuthGate;
