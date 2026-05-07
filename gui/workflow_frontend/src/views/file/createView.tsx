import React, { useState } from 'react';
import {
  Box,
  Button,
  Text,
  Input,
  VStack,
  useToast,
  FormControl,
  FormLabel,
  Textarea,
  Grid,
  GridItem,
  Divider,
  Spinner,
  useColorModeValue,
  Radio,
  RadioGroup,
  Select,
  Stack,
} from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { createAuthHeaders } from '../../api/authHeaders'; // for authentication header
import { WorkflowContextEditor } from '../../components/WorkflowContextEditor';
import type { HpcTarget, Visibility } from '../home/type';

interface CreateFlowProjectRequest {
  name: string;
  description: string;
  workflow_context?: Record<string, any>;
  visibility: Visibility;
  reference: string;
  hpc_target: HpcTarget;
}

interface CreateFlowProjectResponse {
  id: string;  // UUID is a string
  name: string;
  description: string;
  workflow_context?: Record<string, any>;
  created_at: string;
  updated_at: string;
  is_active: boolean;
  owner?: number;
}

interface ErrorResponse {
  error: string;
  details?: string;
}

const CreateFlowPj: React.FC = () => {
  const [projectName, setProjectName] = useState<string>('');
  const [note, setNote] = useState<string>('');
  const [workflowContext, setWorkflowContext] = useState<Record<string, any> | null>(null);
  const [isContextValid, setIsContextValid] = useState<boolean>(true);
  const [contextResetKey, setContextResetKey] = useState<number>(0);
  const [visibility, setVisibility] = useState<Visibility>('private');
  const [reference, setReference] = useState<string>('');
  const [hpcTarget, setHpcTarget] = useState<HpcTarget>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const toast = useToast();
  const navigate = useNavigate();

  const panelBg = useColorModeValue('#f7f7f8', 'gray.900');
  const textColor = useColorModeValue('#1a1a1a', 'white');
  const inputBorder = useColorModeValue('gray.300', 'gray.600');
  const inputHoverBorder = useColorModeValue('blue.500', 'blue.300');
  const inputFocusBorder = useColorModeValue('blue.500', 'blue.300');
  const inputFocusShadow = useColorModeValue('0 0 0 1px #3182ce', '0 0 0 1px #63b3ed');

  // Backend Workflow API endpoints
  const API_ENDPOINT = `/api/workflow/`;

  // Workflow project creation API call
  const createFlowProject = async (data: CreateFlowProjectRequest): Promise<CreateFlowProjectResponse> => {
    try {
      console.log('Creating flow project with data:', data);

      // Get authentication header
      const authHeaders = await createAuthHeaders();

      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        credentials: 'include',  // Include Cookies (Sessions)
        headers: {
          ...authHeaders,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      console.log('Response status:', response.status);
      console.log('Response ok:', response.ok);

      if (!response.ok) {
        const errorData: ErrorResponse = await response.json();
        console.error('API error response:', errorData);
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result: CreateFlowProjectResponse = await response.json();
      console.log('Project created successfully:', result);
      return result;
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  };


  const handleRegistration = async () => {
    if (!projectName.trim()) {
      toast({
        title: 'Input Error',
        description: 'Please enter a project name',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);

    try {
      // Prepare data for API calls
      let workflowContextPayload: Record<string, any> | undefined = undefined;
      if (!isContextValid) {
        toast({
          title: 'Invalid JSON',
          description: 'Workflow context must be valid JSON.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
        setIsLoading(false);
        return;
      }
      if (workflowContext && Object.keys(workflowContext).length > 0) {
        workflowContextPayload = workflowContext;
      }

      const requestData: CreateFlowProjectRequest = {
        name: projectName.trim(),
        description: note.trim() || '', // Default to empty string
        visibility,
        reference,
        hpc_target: hpcTarget,
        ...(workflowContextPayload ? { workflow_context: workflowContextPayload } : {}),
      };

      const response = await createFlowProject(requestData);

      // Processing on success
      toast({
        title: 'Creation Success',
        description: `"${response.name}" has been created successfully`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      // reset form
      setProjectName('');
      setNote('');
      setWorkflowContext(null);
      setIsContextValid(true);
      setContextResetKey(prev => prev + 1);
      setVisibility('private');
      setReference('');
      setHpcTarget('');

      // Go to the details screen of the created workflow (using UUID)
      // navigate(`/workflow/${response.id}`);
      navigate(`/`);

    } catch (error) {
      console.error('Error creating flow project:', error);

      let errorMessage = 'An unexpected error occurred';

      if (error instanceof Error) {
        errorMessage = error.message;
      }

      // Detailed message for the specific error
      if (errorMessage.includes('401')) {
        errorMessage = 'Authentication required. Please login first.';
      } else if (errorMessage.includes('403')) {
        errorMessage = 'Permission denied. You do not have access to create projects.';
      } else if (errorMessage.includes('400')) {
        errorMessage = 'Invalid project data. Please check your input.';
      } else if (errorMessage.includes('500')) {
        errorMessage = 'Server error. Please try again later.';
      }

      toast({
        title: 'Creation Failed',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    setProjectName('');
    setNote('');
    setWorkflowContext(null);
    setIsContextValid(true);
    setContextResetKey(prev => prev + 1);
    setReference('');
    setHpcTarget('');
    navigate(-1);
  };

  return (
    <Box height="100%" width="100%" overflow="auto" bg={panelBg}>
      <VStack spacing={6} width="100%" p={6} maxWidth="600px" mx="auto" minHeight="100vh">
      <Text fontSize="2xl" fontWeight="bold" mb={2} color={textColor}>
        🚀 Create Flow Project
      </Text>

      <Text fontSize="md" color={textColor} textAlign="center" mb={2}>
        Create a new workflow project to design mathematical calculations
      </Text>

      <Divider my={4} />

      <VStack width="100%" spacing={5} align="start">
        <FormControl isRequired>
          <FormLabel htmlFor="projectName" fontSize="sm" fontWeight="semibold" color={textColor}>
            Project Name
          </FormLabel>
          <Input
            id="projectName"
            placeholder="Enter your project name..."
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            isDisabled={isLoading}
            size="md"
            borderColor={inputBorder}
            _hover={{ borderColor: inputHoverBorder }}
            _focus={{ borderColor: inputFocusBorder, boxShadow: inputFocusShadow }}
          />
        </FormControl>

        <FormControl>
          <FormLabel htmlFor="note" fontSize="sm" fontWeight="semibold" color={textColor}>
            Description (Optional)
          </FormLabel>
          <Textarea
            id="note"
            placeholder="Describe your workflow project..."
            rows={4}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            isDisabled={isLoading}
            borderColor={inputBorder}
            _hover={{ borderColor: inputHoverBorder }}
            _focus={{ borderColor: inputFocusBorder, boxShadow: inputFocusShadow }}
            resize="vertical"
          />
        </FormControl>

        <FormControl>
          <FormLabel fontSize="sm" fontWeight="semibold" color={textColor}>
            Visibility
          </FormLabel>
          <RadioGroup
            value={visibility}
            onChange={(value) => setVisibility(value as Visibility)}
            isDisabled={isLoading}
          >
            <Stack direction="row" spacing={6}>
              <Radio value="private">
                Private
                <Text as="span" fontSize="xs" color="gray.500" ml={2}>
                  Only you can view, edit, and run
                </Text>
              </Radio>
              <Radio value="public">
                Public
                <Text as="span" fontSize="xs" color="gray.500" ml={2}>
                  Any signed-in user can view, edit, and run
                </Text>
              </Radio>
            </Stack>
          </RadioGroup>
        </FormControl>

        <FormControl>
          <FormLabel htmlFor="reference" fontSize="sm" fontWeight="semibold" color={textColor}>
            Reference (Optional)
          </FormLabel>
          <Textarea
            id="reference"
            placeholder="Papers, URLs, or other references for this workflow..."
            rows={3}
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            isDisabled={isLoading}
            borderColor={inputBorder}
            _hover={{ borderColor: inputHoverBorder }}
            _focus={{ borderColor: inputFocusBorder, boxShadow: inputFocusShadow }}
            resize="vertical"
          />
        </FormControl>

        <FormControl>
          <FormLabel htmlFor="hpcTarget" fontSize="sm" fontWeight="semibold" color={textColor}>
            HPC Target (Optional)
          </FormLabel>
          <Select
            id="hpcTarget"
            value={hpcTarget}
            onChange={(e) => setHpcTarget(e.target.value as HpcTarget)}
            isDisabled={isLoading}
            borderColor={inputBorder}
            _hover={{ borderColor: inputHoverBorder }}
            _focus={{ borderColor: inputFocusBorder, boxShadow: inputFocusShadow }}
          >
            <option value="">Not specified</option>
            <option value="riken">Riken</option>
            <option value="fugaku">Fugaku</option>
          </Select>
        </FormControl>

        <FormControl>
          <WorkflowContextEditor
            key={contextResetKey}
            label="Workflow Context (Optional)"
            disabled={isLoading}
            onChange={(context, rawText, isValid) => {
              setWorkflowContext(context);
              setIsContextValid(isValid);
            }}
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
            isDisabled={!projectName.trim() || isLoading}
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
            leftIcon={isLoading ? <Spinner size="sm" /> : undefined}
          >
            {isLoading ? 'Creating Project...' : 'Create Project'}
          </Button>
        </GridItem>
        <GridItem>
          <Button
            colorScheme="red"
            variant="outline"
            size="lg"
            width="100%"
            onClick={handleCancel}
            _hover={{
              bg: "red.50",
              borderColor: "red.300",
              color: "red.600"
            }}
            isDisabled={isLoading}
          >
            Cancel
          </Button>
        </GridItem>
      </Grid>
      </VStack>
    </Box>
  );
};

export default CreateFlowPj;
