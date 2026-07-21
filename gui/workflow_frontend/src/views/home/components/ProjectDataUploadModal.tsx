import { useCallback, useEffect, useRef, useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Flex,
  FormLabel,
  HStack,
  IconButton,
  Input,
  List,
  ListItem,
  Modal,
  ModalBody,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Progress,
  Text,
  VStack,
  useToast,
} from "@chakra-ui/react";
import { AttachmentIcon, DeleteIcon } from "@chakra-ui/icons";
import {
  PROJECT_UPLOAD_MAX_BYTES,
  deleteProjectFile,
  formatBytes,
  listProjectFiles,
  ProjectFileInfo,
  uploadProjectFiles,
} from "../../../api/projectFilesApi";

type Props = {
  projectId: string | null;
  projectName?: string;
  isOpen: boolean;
  onClose: () => void;
};

export function ProjectDataUploadModal({
  projectId,
  projectName,
  isOpen,
  onClose,
}: Props) {
  const toast = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<ProjectFileInfo[]>([]);
  const [maxBytes, setMaxBytes] = useState(PROJECT_UPLOAD_MAX_BYTES);
  const [selected, setSelected] = useState<File[]>([]);
  const [overwrite, setOverwrite] = useState(false);
  const [loadingList, setLoadingList] = useState(false);
  const [uploading, setUploading] = useState(false);

  const refresh = useCallback(async () => {
    if (!projectId) return;
    setLoadingList(true);
    try {
      const data = await listProjectFiles(projectId);
      setFiles(data.files || []);
      if (data.max_bytes) setMaxBytes(data.max_bytes);
    } catch (err: any) {
      toast({
        title: "Could not list project files",
        description: err?.message || String(err),
        status: "error",
        duration: 4000,
        isClosable: true,
      });
    } finally {
      setLoadingList(false);
    }
  }, [projectId, toast]);

  useEffect(() => {
    if (isOpen && projectId) {
      setSelected([]);
      setOverwrite(false);
      void refresh();
    }
  }, [isOpen, projectId, refresh]);

  const onPickFiles = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    const next: File[] = [];
    for (let i = 0; i < list.length; i++) {
      const file = list.item(i);
      if (!file) continue;
      if (file.size > maxBytes) {
        toast({
          title: "File too large",
          description: `${file.name} exceeds ${formatBytes(maxBytes)}`,
          status: "warning",
          duration: 4000,
          isClosable: true,
        });
        continue;
      }
      next.push(file);
    }
    setSelected((prev) => [...prev, ...next]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleUpload = async () => {
    if (!projectId || selected.length === 0) return;
    setUploading(true);
    try {
      const result = await uploadProjectFiles(projectId, selected, { overwrite });
      const okCount = result.uploaded?.length || 0;
      const errCount = result.errors?.length || 0;
      if (okCount > 0) {
        toast({
          title: errCount ? "Upload partially completed" : "Upload complete",
          description: `${okCount} file(s) saved to the project folder${
            errCount ? `; ${errCount} failed` : ""
          }.`,
          status: errCount ? "warning" : "success",
          duration: 4000,
          isClosable: true,
        });
      }
      if (errCount && okCount === 0) {
        toast({
          title: "Upload failed",
          description: result.errors.map((e) => e.error).join("; "),
          status: "error",
          duration: 5000,
          isClosable: true,
        });
      } else if (errCount) {
        toast({
          title: "Some files failed",
          description: result.errors.map((e) => `${e.filename}: ${e.error}`).join("; "),
          status: "warning",
          duration: 5000,
          isClosable: true,
        });
      }
      setSelected([]);
      await refresh();
    } catch (err: any) {
      toast({
        title: "Upload failed",
        description: err?.message || String(err),
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (filename: string) => {
    if (!projectId) return;
    try {
      await deleteProjectFile(projectId, filename);
      toast({
        title: "File deleted",
        description: filename,
        status: "success",
        duration: 2000,
        isClosable: true,
      });
      await refresh();
    } catch (err: any) {
      toast({
        title: "Delete failed",
        description: err?.message || String(err),
        status: "error",
        duration: 4000,
        isClosable: true,
      });
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="lg">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>
          Upload data to project
          {projectName ? (
            <Text as="span" fontWeight="normal" fontSize="md" color="gray.600">
              {" "}
              — {projectName}
            </Text>
          ) : null}
        </ModalHeader>
        <ModalBody>
          <Text fontSize="sm" color="gray.600" mb={3}>
            Files are saved into this project&apos;s folder (
            <Text as="span" fontFamily="mono" fontSize="xs">
              codes/projects/&lt;id&gt;/
            </Text>
            ), the same place JupyterLab opens. Max size per file:{" "}
            {formatBytes(maxBytes)}.
          </Text>

          <FormLabel fontSize="sm">Select files</FormLabel>
          <Input
            ref={fileInputRef}
            type="file"
            multiple
            display="none"
            onChange={(e) => onPickFiles(e.target.files)}
          />
          <Button
            leftIcon={<AttachmentIcon />}
            onClick={() => fileInputRef.current?.click()}
            variant="outline"
            size="sm"
            mb={2}
          >
            Choose files
          </Button>

          {selected.length > 0 && (
            <Box mb={3} p={2} bg="gray.50" borderRadius="md">
              <Text fontSize="xs" fontWeight="semibold" mb={1}>
                Ready to upload ({selected.length})
              </Text>
              <List spacing={1}>
                {selected.map((f) => (
                  <ListItem key={`${f.name}-${f.size}`} fontSize="sm">
                    {f.name}{" "}
                    <Text as="span" color="gray.500">
                      ({formatBytes(f.size)})
                    </Text>
                  </ListItem>
                ))}
              </List>
              <Checkbox
                mt={2}
                isChecked={overwrite}
                onChange={(e) => setOverwrite(e.target.checked)}
                size="sm"
              >
                Overwrite existing files with the same name
              </Checkbox>
            </Box>
          )}

          {uploading && <Progress size="xs" isIndeterminate mb={3} />}

          <Flex justify="space-between" align="center" mb={1}>
            <FormLabel fontSize="sm" mb={0}>
              Files in project folder
            </FormLabel>
            <Button size="xs" variant="ghost" onClick={() => void refresh()} isLoading={loadingList}>
              Refresh
            </Button>
          </Flex>
          {files.length === 0 ? (
            <Text fontSize="sm" color="gray.500">
              {loadingList ? "Loading…" : "No files yet."}
            </Text>
          ) : (
            <VStack align="stretch" spacing={1} maxH="220px" overflowY="auto">
              {files.map((f) => (
                <HStack
                  key={f.filename}
                  justify="space-between"
                  px={2}
                  py={1}
                  borderWidth={1}
                  borderColor="gray.100"
                  borderRadius="md"
                >
                  <Box>
                    <Text fontSize="sm">{f.filename}</Text>
                    <Text fontSize="xs" color="gray.500">
                      {formatBytes(f.size_bytes)}
                    </Text>
                  </Box>
                  <IconButton
                    aria-label={`Delete ${f.filename}`}
                    icon={<DeleteIcon />}
                    size="xs"
                    variant="ghost"
                    colorScheme="red"
                    onClick={() => void handleDelete(f.filename)}
                  />
                </HStack>
              ))}
            </VStack>
          )}
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose} isDisabled={uploading}>
            Close
          </Button>
          <Button
            colorScheme="blue"
            onClick={() => void handleUpload()}
            isLoading={uploading}
            isDisabled={selected.length === 0}
          >
            Upload
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
