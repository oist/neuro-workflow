import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  Flex,
  Text,
  Input,
  VStack,
  useToast,
  HStack,
  Icon,
  List,
  ListItem,
  FormControl,
  FormLabel,
  Textarea,
  Grid,
  GridItem,
  Divider,
  Progress,
  Spinner,
  Select,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import { AttachmentIcon, CloseIcon, CheckIcon } from '@chakra-ui/icons';
import { useNavigate } from 'react-router-dom';
import { useCallback } from "react";

// Category definition
let categories = {
  analysis: { label: 'Analysis', color: 'blue', description: 'Data analysis and statistics' },
  io: { label: 'I/O', color: 'green', description: 'Input/output operations' },
  network: { label: 'Network', color: 'purple', description: 'Network and communication' },
  optimization: { label: 'Optimization', color: 'orange', description: 'Optimization algorithms' },
  simulation: { label: 'Simulation', color: 'red', description: 'Simulation and modeling' },
  stimulus: { label: 'Stimulus', color: 'teal', description: 'Stimulus generation and control' }
};

type CategoryKey = keyof typeof categories;

/*
const CATEGORIES = {
  analysis: { label: 'Analysis', color: 'blue', description: 'Data analysis and statistics' },
  io: { label: 'I/O', color: 'green', description: 'Input/output operations' },
  network: { label: 'Network', color: 'purple', description: 'Network and communication' },
  optimization: { label: 'Optimization', color: 'orange', description: 'Optimization algorithms' },
  simulation: { label: 'Simulation', color: 'red', description: 'Simulation and modeling' },
  stimulus: { label: 'Stimulus', color: 'teal', description: 'Stimulus generation and control' }
} as const;

type CategoryKey = keyof typeof CATEGORIES;
*/

// API response type definition
interface UploadResponse {
  id: string;
  name: string;
  description: string;
  file: string;
  uploaded_by: string | null;
  uploaded_by_name: string | null;
  file_size: number;
  is_analyzed: boolean;
  analysis_error: string | null;
  node_classes_count: number;
  category?: CategoryKey;
  created_at: string;
  updated_at: string;
}

interface UploadError {
  error: string;
  [key: string]: any;
}

const BoxUpload: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [category, setCategory] = useState<CategoryKey>('analysis');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadedFiles, setUploadedFiles] = useState<UploadResponse[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();
  const navigate = useNavigate();

  const panelBg = useColorModeValue('#f7f7f8', 'gray.900');
  const textColor = useColorModeValue('#1a1a1a', 'white');
  const subtextColor = useColorModeValue('gray.500', 'gray.300');
  const borderColor = useColorModeValue('#e5e5e5', 'gray.600');
  const inputBg = useColorModeValue('white', 'gray.700');

  useEffect(() => {
    if (selectedFiles.length > 0 && fileName === '') {
      const fullName = selectedFiles[0].name;
      const nameWithoutExtension = fullName.replace(/\.py$/, '');
      setFileName(nameWithoutExtension);
    }
  }, [selectedFiles, fileName]);

  // Categories interface
  interface UseCategoriesReturn {
    data: UploadedNodesResponse | null;
    isLoading: boolean;
    error: string | null;
    refetch: () => Promise<void>;
  }

  // Get Categories
  const getCategories = (): UseUploadedNodesReturn => {
    const [data, setData] = useState<UploadedNodesResponse | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchUploadedNodes = useCallback(async () => {
      try {
        setIsLoading(true);
        setError(null);

        const response = await fetch("/api/box/categories/");

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        //const result: UploadedNodesResponse = await response.json();
        let catJson = await response.json();
        for (const key in catJson.categories) {
          if (!(catJson.categories[key].value in categories)){
            categories[catJson.categories[key].value] = {
              label: catJson.categories[key].label,
              color: catJson.categories[key].settings['color'],
              description: "",
            };
          }
        }

        console.log("This is response data", categories);

        setData(categories);
      } catch (err) {
        console.error("Failed to fetch uploaded nodes:", err);
        setError(err instanceof Error ? err.message : "Failed to fetch nodes");
        setData(null);
      } finally {
        setIsLoading(false);
      }
    }, []);

    useEffect(() => {
      fetchUploadedNodes();
    }, [fetchUploadedNodes]);

    return {
      data,
      isLoading,
      error,
      refetch: fetchUploadedNodes,
    };
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newFiles: File[] = [];
    const rejectedFiles: { name: string; reason: string }[] = [];

    Array.from(files).forEach((file) => {
      if (!file.name.endsWith('.py')) {
        rejectedFiles.push({
          name: file.name,
          reason: 'Only Python files (.py) are allowed',
        });
        return;
      }

      // 10MB limit check
      if (file.size > 10 * 1024 * 1024) {
        rejectedFiles.push({
          name: file.name,
          reason: 'File size must be less than 10MB',
        });
        return;
      }

      newFiles.push(file);
    });

    if (rejectedFiles.length > 0) {
      rejectedFiles.forEach((file) => {
        toast({
          title: 'File rejected',
          description: `${file.name}: ${file.reason}`,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      });
    }

    if (newFiles.length > 0) {
      setSelectedFiles((prev) => [...prev, ...newFiles]);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));

    if (selectedFiles.length === 1) {
      setFileName('');
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = e.dataTransfer.files;

      const newFiles: File[] = [];
      const rejectedFiles: { name: string; reason: string }[] = [];

      Array.from(files).forEach((file) => {
        if (!file.name.endsWith('.py')) {
          rejectedFiles.push({
            name: file.name,
            reason: 'Only Python files (.py) are allowed',
          });
          return;
        }

        if (file.size > 10 * 1024 * 1024) {
          rejectedFiles.push({
            name: file.name,
            reason: 'File size must be less than 10MB',
          });
          return;
        }

        newFiles.push(file);
      });

      if (rejectedFiles.length > 0) {
        rejectedFiles.forEach((file) => {
          toast({
            title: 'File rejected',
            description: `${file.name}: ${file.reason}`,
            status: 'error',
            duration: 5000,
            isClosable: true,
          });
        });
      }

      if (newFiles.length > 0) {
        setSelectedFiles((prev) => [...prev, ...newFiles]);
      }
    }
  };

  const formatFileSize = (sizeInBytes: number): string => {
    if (sizeInBytes < 1024) {
      return `${sizeInBytes} B`;
    } else if (sizeInBytes < 1024 * 1024) {
      return `${(sizeInBytes / 1024).toFixed(2)} KB`;
    } else {
      return `${(sizeInBytes / (1024 * 1024)).toFixed(2)} MB`;
    }
  };

  const uploadFile = async (file: File, name?: string, desc?: string, cat?: CategoryKey): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (name) formData.append('name', name);
    if (desc) formData.append('description', desc);
    if (cat) formData.append('category', cat);

    const response = await fetch('/api/box/upload/', {
      method: 'POST',
      body: formData,
      // Note: Do not specify the Content-Type header as it is set automatically.
    });

    if (!response.ok) {
      const errorData: UploadError = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  };

  const handleRegistration = async () => {
    // Input validation
    if (selectedFiles.length === 0) {
      toast({
        title: 'Input Error',
        description: 'Please select your Python files',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!fileName.trim()) {
      toast({
        title: 'Input Error',
        description: 'Please enter a file name',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    const newUploadedFiles: UploadResponse[] = [];

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        const fileNameToUse = selectedFiles.length === 1 ? fileName : file.name.replace(/\.py$/, '');

        try {
          const result = await uploadFile(file, fileNameToUse, description, category);
          newUploadedFiles.push(result);

          // Progress update
          setUploadProgress(((i + 1) / selectedFiles.length) * 100);

          toast({
            title: 'Upload Success',
            description: `${result.name} has been uploaded successfully!`,
            status: 'success',
            duration: 3000,
            isClosable: true,
          });

        } catch (error) {
          console.error(`Failed to upload ${file.name}:`, error);
          toast({
            title: 'Upload Failed',
            description: `Failed to upload ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
            status: 'error',
            duration: 5000,
            isClosable: true,
          });
        }
      }

      if (newUploadedFiles.length > 0) {
        setUploadedFiles(prev => [...prev, ...newUploadedFiles]);

        // reset form
        setSelectedFiles([]);
        setFileName('');
        setDescription('');
        setCategory('analysis');
      }

    } catch (error) {
      console.error('Upload process failed:', error);
      toast({
        title: 'Upload Failed',
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleCancel = () => {
    setSelectedFiles([]);
    setFileName('');
    setDescription('');
    setCategory('analysis');
    setUploadedFiles([]);
    navigate(-1);
  };

  const catlist = getCategories();

  return (
    <Box height="100%" width="100%" overflow="auto" bg={panelBg}>
      <VStack spacing={6} width="100%" p={6} maxWidth="600px" mx="auto" minHeight="100vh">
      <Text fontSize="2xl" fontWeight="bold" mb={2} color={textColor}>
        📁 Python File Upload
      </Text>

      <Text fontSize="md" color={textColor} textAlign="center" mb={2}>
        Upload and analyze your Python files for code visualization
      </Text>

      <Divider my={4} />

      <Flex
        direction="column"
        align="center"
        justify="center"
        p={8}
        borderWidth={2}
        borderRadius="lg"
        borderStyle="dashed"
        borderColor="blue.400"
        bg="rgba(59, 130, 246, 0.1)"
        width="100%"
        minHeight="200px"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        cursor="pointer"
        onClick={() => fileInputRef.current?.click()}
        _hover={{
          bg: "rgba(59, 130, 246, 0.2)",
          borderColor: "blue.300",
          transform: "translateY(-2px)"
        }}
        transition="all 0.2s"
      >
        <Input
          type="file"
          multiple
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".py"
          display="none"
        />
        <Icon as={AttachmentIcon} w={12} h={12} color="blue.400" mb={4} />
        <Text fontWeight="bold" fontSize="xl" mb={3} color={textColor}>
          Drop Python files here
        </Text>
        <Text fontSize="md" color={subtextColor}>
          Or click to select files
        </Text>
        <Text fontSize="sm" color={subtextColor} mt={2}>
          .py files only (max 10MB per file)
        </Text>
      </Flex>

      {selectedFiles.length > 0 && (
        <Box width="100%" border="1px" borderColor={borderColor} borderRadius="md" p={4} bg="rgba(255, 255, 255, 0.05)">
          <Text fontWeight="semibold" mb={3} color={textColor}>
            Selected files ({selectedFiles.length})
          </Text>
          <List spacing={2} width="100%">
            {selectedFiles.map((file, index) => (
              <ListItem key={index}>
                <HStack
                  p={3}
                  bg="rgba(255, 255, 255, 0.1)"
                  borderRadius="md"
                  justifyContent="space-between"
                  borderLeft="4px"
                  borderLeftColor="blue.400"
                  _hover={{ bg: "rgba(255, 255, 255, 0.15)" }}
                  transition="all 0.2s"
                >
                  <HStack>
                    <Text fontWeight="medium" color={textColor}>
                      {file.name}
                    </Text>
                    <Text fontSize="sm" color={subtextColor}>
                      ({formatFileSize(file.size)})
                    </Text>
                  </HStack>
                  <Button
                    size="sm"
                    variant="ghost"
                    colorScheme="red"
                    onClick={() => removeFile(index)}
                    disabled={isUploading}
                    _hover={{ bg: "red.500", color: "white" }}
                  >
                    <Icon as={CloseIcon} />
                  </Button>
                </HStack>
              </ListItem>
            ))}
          </List>
        </Box>
      )}

      {isUploading && (
        <Box width="100%" p={4} bg="rgba(59, 130, 246, 0.1)" borderRadius="md" border="1px" borderColor="blue.400">
          <HStack mb={2}>
            <Spinner size="sm" color="blue.400" />
            <Text fontWeight="semibold" color={textColor}>Uploading files...</Text>
          </HStack>
          <Progress value={uploadProgress} colorScheme="blue" size="lg" bg="rgba(255, 255, 255, 0.1)" />
          <Text fontSize="sm" color={subtextColor} mt={2}>
            {Math.round(uploadProgress)}% complete
          </Text>
        </Box>
      )}

      {uploadedFiles.length > 0 && (
        <Box width="100%" border="1px" borderColor="green.400" borderRadius="md" p={4} bg="rgba(34, 197, 94, 0.1)">
          <Text fontWeight="semibold" mb={3} color={textColor}>
            Successfully uploaded files ({uploadedFiles.length})
          </Text>
          <List spacing={2} width="100%">
            {uploadedFiles.map((file, index) => (
              <ListItem key={index}>
                <HStack
                  p={3}
                  bg="rgba(255, 255, 255, 0.1)"
                  borderRadius="md"
                  justifyContent="space-between"
                  borderLeft="4px"
                  borderLeftColor="green.400"
                  _hover={{ bg: "rgba(255, 255, 255, 0.15)" }}
                  transition="all 0.2s"
                >
                  <HStack>
                    <Icon as={CheckIcon} color="green.400" />
                    <Box flex="1">
                      <HStack spacing={2} mb={1}>
                        <Text fontWeight="medium" color={textColor}>
                          {file.name}
                        </Text>
                        {file.category && (
                          <Badge
                            colorScheme={categories[file.category].color}
                            variant="subtle"
                            size="sm"
                          >
                            {categories[file.category].label}
                          </Badge>
                        )}
                      </HStack>
                      <Text fontSize="sm" color={subtextColor}>
                        {file.is_analyzed ? (
                          `Analyzed: ${file.node_classes_count} node classes found`
                        ) : file.analysis_error ? (
                          `Analysis failed: ${file.analysis_error}`
                        ) : (
                          'Analysis in progress...'
                        )}
                      </Text>
                    </Box>
                  </HStack>
                  <Text fontSize="sm" color={subtextColor}>
                    {formatFileSize(file.file_size)}
                  </Text>
                </HStack>
              </ListItem>
            ))}
          </List>
        </Box>
      )}

      <Divider my={4} />

      <VStack width="100%" spacing={5} align="start">
        <FormControl isRequired>
          <FormLabel htmlFor="fileName" fontSize="sm" fontWeight="semibold" color={textColor}>
            File Name
          </FormLabel>
          <Input
            id="fileName"
            placeholder="Enter file name (auto-filled from file)..."
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            disabled={isUploading}
            borderColor="gray.300"
            _hover={{ borderColor: "blue.300" }}
            _focus={{ borderColor: "blue.500", boxShadow: "0 0 0 1px #3182ce" }}
          />
          {selectedFiles.length > 1 && (
            <Text fontSize="sm" color={subtextColor} mt={1}>
              When multiple files are selected, individual file names will be used
            </Text>
          )}
        </FormControl>

        <FormControl isRequired>
          <FormLabel htmlFor="category" fontSize="sm" fontWeight="semibold" color={textColor}>
            Category
          </FormLabel>
          <Select
            id="category"
            value={category}
            onChange={(e) => setCategory(e.target.value as CategoryKey)}
            disabled={isUploading}
            borderColor="gray.300"
            _hover={{ borderColor: "blue.300" }}
            _focus={{ borderColor: "blue.500", boxShadow: "0 0 0 1px #3182ce" }}
            bg={inputBg}
            color="gray.800"
          >
            {Object.entries(categories).map(([key, value]) => (
              <option key={key} value={key} style={{ color: '#2D3748' }}>
                {value.label} - {value.description}
              </option>
            ))}
          </Select>
          <HStack mt={2} spacing={2}>
            <Text fontSize="xs" color={subtextColor}>Selected:</Text>
            <Badge
              colorScheme={categories[category].color}
              variant="subtle"
              fontSize="xs"
            >
              {categories[category].label}
            </Badge>
          </HStack>
        </FormControl>

        <FormControl>
          <FormLabel htmlFor="description" fontSize="sm" fontWeight="semibold" color={textColor}>
            Description (Optional)
          </FormLabel>
          <Textarea
            id="description"
            placeholder="Describe your Python files..."
            rows={4}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={isUploading}
            borderColor="gray.300"
            _hover={{ borderColor: "blue.300" }}
            _focus={{ borderColor: "blue.500", boxShadow: "0 0 0 1px #3182ce" }}
            resize="vertical"
          />
        </FormControl>
      </VStack>

      <Grid templateColumns="repeat(2, 1fr)" gap={4} width="100%" mt={6}>
        <GridItem>
          <Button
            colorScheme="green"
            variant="solid"
            size="lg"
            width="100%"
            fontWeight="bold"
            isDisabled={selectedFiles.length === 0 || !fileName.trim() || isUploading}
            onClick={handleRegistration}
            boxShadow="sm"
            color="white"
            _hover={{
              boxShadow: "md",
              transform: "translateY(-2px)",
              bg: "green.600",
              color: "white"
            }}
            _active={{ transform: "translateY(0)" }}
            transition="all 0.2s"
            leftIcon={isUploading ? <Spinner size="sm" /> : undefined}
          >
            {isUploading ? 'Uploading & Analyzing...' : 'Upload & Analyze'}
          </Button>
        </GridItem>
        <GridItem>
          <Button
            colorScheme="red"
            variant="outline"
            size="lg"
            width="100%"
            onClick={handleCancel}
            disabled={isUploading}
            _hover={{
              bg: "red.50",
              borderColor: "red.300",
              color: "red.600"
            }}
          >
            Cancel
          </Button>
        </GridItem>
      </Grid>
      </VStack>
    </Box>
  );
};

export default BoxUpload;
