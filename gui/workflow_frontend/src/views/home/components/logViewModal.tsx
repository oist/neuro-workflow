import { useRef, useEffect } from "react";
import {
  Button,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Box,
  Badge,
  HStack,
  Image,
  Spinner,
  Text,
  useColorModeValue,
} from "@chakra-ui/react";

export interface LogEntry {
  type: "stdout" | "stderr" | "execute_result" | "error" | "info" | "image";
  content: string;
  mime?: string; // set for type "image" (e.g. "image/png")
}

interface LogViewProps {
  isOpen: boolean;
  onClose: () => void;
  logEntries: LogEntry[];
  isRunning: boolean;
  runStatus: "idle" | "running" | "ok" | "error";
  onStop?: () => void;
}

const typeStyles: Record<string, { color: string; bg: string }> = {
  stdout: { color: "green.200", bg: "transparent" },
  stderr: { color: "yellow.300", bg: "transparent" },
  execute_result: { color: "cyan.200", bg: "transparent" },
  error: { color: "red.300", bg: "red.900" },
  info: { color: "gray.400", bg: "transparent" },
};

export default function LogViewModal({
  isOpen,
  onClose,
  logEntries,
  isRunning,
  runStatus,
  onStop,
}: LogViewProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  const panelBg = useColorModeValue('#f7f7f8', 'gray.900');
  const subtextColor = useColorModeValue('gray.500', 'gray.500');
  const logTextColor = useColorModeValue('gray.800', 'gray.100');

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logEntries]);

  const statusBadge = () => {
    switch (runStatus) {
      case "running":
        return (
          <Badge colorScheme="blue" variant="subtle">
            <HStack spacing={1}>
              <Spinner size="xs" />
              <Text fontSize="xs">Running</Text>
            </HStack>
          </Badge>
        );
      case "ok":
        return <Badge colorScheme="green">Completed</Badge>;
      case "error":
        return <Badge colorScheme="red">Error</Badge>;
      default:
        return null;
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="6xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>
          <HStack spacing={3}>
            <Text>Workflow Execution</Text>
            {statusBadge()}
          </HStack>
        </ModalHeader>

        <ModalBody>
          <Box
            bg={panelBg}
            color={logTextColor}
            fontFamily="mono"
            fontSize="sm"
            p={4}
            borderRadius="md"
            height="500px"
            overflowY="auto"
            whiteSpace="pre-wrap"
            wordBreak="break-all"
          >
            {logEntries.length === 0 && isRunning && (
              <Text color={subtextColor}>Waiting for output...</Text>
            )}
            {logEntries.map((entry, i) => {
              if (entry.type === "image") {
                return (
                  <Image
                    key={i}
                    src={`data:${entry.mime || "image/png"};base64,${entry.content}`}
                    alt="figure output"
                    maxW="100%"
                    maxH="400px"
                    my={2}
                    borderRadius="md"
                  />
                );
              }
              const style = typeStyles[entry.type] || typeStyles.info;
              return (
                <Box
                  key={i}
                  color={style.color}
                  bg={style.bg}
                  px={style.bg !== "transparent" ? 2 : 0}
                  py={style.bg !== "transparent" ? 1 : 0}
                  borderRadius={style.bg !== "transparent" ? "sm" : undefined}
                >
                  {entry.content}
                </Box>
              );
            })}
            <div ref={bottomRef} />
          </Box>
        </ModalBody>

        <ModalFooter>
          {isRunning && onStop && (
            <Button colorScheme="red" variant="outline" mr={3} onClick={onStop}>
              Stop
            </Button>
          )}
          <Button onClick={onClose}>Close</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
