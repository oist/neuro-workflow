import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Button,
  VStack,
  HStack,
  Text,
  Input,
  Textarea,
  FormControl,
  FormLabel,
  FormHelperText,
  useToast,
  Spinner,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Box,
  Badge,
  Select,
  Code,
  Divider,
  Checkbox,
  Collapse,
} from '@chakra-ui/react';
import { ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons';
import { CheckIcon, CloseIcon } from '@chakra-ui/icons';
import { createAuthHeaders } from '../../../api/authHeaders';

interface CustomDatabase {
  id: string;
  name: string;
  description?: string;
  base_url: string;
  api_key?: string;
  adapter_type: string;
  is_active: boolean;
  is_verified: boolean;
  last_tested?: string;
  test_result?: string;
  test_error?: string;
  config?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

interface CustomDatabaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  database?: CustomDatabase | null; // If provided, edit mode; otherwise, create mode
  onSuccess?: () => void;
}

const CustomDatabaseModal: React.FC<CustomDatabaseModalProps> = ({
  isOpen,
  onClose,
  database,
  onSuccess,
}) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    base_url: '',
    api_key: '',
    adapter_type: 'rest_api',
    is_active: true,
    config: {} as Record<string, any>,
  });
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [saving, setSaving] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [queryEndpoint, setQueryEndpoint] = useState('');
  const [authType, setAuthType] = useState('api_key');
  const [apiKeyHeader, setApiKeyHeader] = useState('X-API-Key');
  const [customConfigJson, setCustomConfigJson] = useState('');
  const [apiKeySecretName, setApiKeySecretName] = useState('');
  const toast = useToast();

  const isEditMode = !!database;

  // Load database data when in edit mode
  useEffect(() => {
    if (isOpen && database) {
      const config = database.config || {};
      setFormData({
        name: database.name || '',
        description: database.description || '',
        base_url: database.base_url || '',
        api_key: database.api_key || '',
        adapter_type: database.adapter_type || 'rest_api',
        is_active: database.is_active ?? true,
        config: config,
      });
      // Load advanced settings
      setQueryEndpoint(config.query_endpoint || '');
      setAuthType(config.auth_type || 'api_key');
      setApiKeyHeader(config.api_key_header || 'X-API-Key');
      setCustomConfigJson(JSON.stringify(config, null, 2));
      setApiKeySecretName(config.api_key_secret?.__nw_secret?.name || '');
      setTestResult(null);
    } else if (isOpen && !database) {
      // Reset form for create mode
      setFormData({
        name: '',
        description: '',
        base_url: '',
        api_key: '',
        adapter_type: 'rest_api',
        is_active: true,
        config: {},
      });
      // Reset advanced settings
      setQueryEndpoint('');
      setAuthType('api_key');
      setApiKeyHeader('X-API-Key');
      setCustomConfigJson('');
      setApiKeySecretName('');
      setTestResult(null);
      setShowAdvanced(false);
    }
  }, [isOpen, database]);

  const buildConfig = () => {
    const config: Record<string, any> = {};
    
    // Add basic config fields if provided
    if (queryEndpoint) {
      config.query_endpoint = queryEndpoint;
    }
    if (authType) {
      config.auth_type = authType;
    }
    if (apiKeyHeader && authType === 'api_key') {
      config.api_key_header = apiKeyHeader;
    }
    
    // Merge with custom JSON config if provided
    if (customConfigJson.trim()) {
      try {
        const customConfig = JSON.parse(customConfigJson);
        Object.assign(config, customConfig);
      } catch (e) {
        // Invalid JSON, will be caught in validation
      }
    }
    
    return config;
  };

  const handleTestConnection = async () => {
    if (!formData.base_url) {
      toast({
        title: 'Error',
        description: 'Please provide a base URL',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setTesting(true);
    setTestResult(null);

    const config = buildConfig();

    try {
      const response = await fetch('/api/metadata/custom-databases/test-connection/', {
        method: 'POST',
        headers: {
          ...await createAuthHeaders(),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          base_url: formData.base_url,
          api_key: apiKeySecretName ? undefined : (formData.api_key || undefined),
          api_key_secret_name: apiKeySecretName || undefined,
          config: config,
        }),
      });

      const data = await response.json();
      setTestResult(data);

      if (data.success) {
        toast({
          title: 'Connection Successful',
          description: data.message,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Connection Failed',
          description: data.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to test connection';
      setTestResult({
        success: false,
        error: 'Request failed',
        message: errorMessage,
      });
      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    if (!formData.name || !formData.base_url) {
      toast({
        title: 'Validation Error',
        description: 'Name and Base URL are required',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    // Validate custom JSON config if provided
    if (customConfigJson.trim()) {
      try {
        JSON.parse(customConfigJson);
      } catch (e) {
        toast({
          title: 'Invalid JSON',
          description: 'Custom configuration JSON is invalid. Please check the syntax.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        return;
      }
    }

    setSaving(true);

    const config = buildConfig();

    try {
      const url = isEditMode
        ? `/api/metadata/custom-databases/${database.id}/`
        : '/api/metadata/custom-databases/';

      const method = isEditMode ? 'PUT' : 'POST';

      const response = await fetch(url, {
        method,
        headers: {
          ...await createAuthHeaders(),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          description: formData.description || undefined,
          base_url: formData.base_url,
          api_key: formData.api_key || undefined,
          api_key_secret_name: apiKeySecretName || undefined,
          adapter_type: formData.adapter_type,
          is_active: formData.is_active,
          config: config,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
        throw new Error(errorData.message || errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      toast({
        title: isEditMode ? 'Database Updated' : 'Database Created',
        description: `Custom database "${formData.name}" ${isEditMode ? 'updated' : 'created'} successfully`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      if (onSuccess) {
        onSuccess();
      }
      onClose();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to save database';
      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!database || !isEditMode) return;

    if (!window.confirm(`Are you sure you want to delete "${database.name}"?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/metadata/custom-databases/${database.id}/`, {
        method: 'DELETE',
        headers: await createAuthHeaders(),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
        throw new Error(errorData.message || errorData.error || `HTTP error! status: ${response.status}`);
      }

      toast({
        title: 'Database Deleted',
        description: `Custom database "${database.name}" deleted successfully`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      if (onSuccess) {
        onSuccess();
      }
      onClose();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete database';
      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl" scrollBehavior="inside">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>
          {isEditMode ? 'Edit Custom Database' : 'Add Custom Database'}
        </ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4} align="stretch">
            <FormControl isRequired>
              <FormLabel>Name</FormLabel>
              <Input
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="My Custom Database"
              />
              <FormHelperText>A friendly name for this database</FormHelperText>
            </FormControl>

            <FormControl>
              <FormLabel>Description</FormLabel>
              <Textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="A description of this database..."
                rows={2}
              />
            </FormControl>

            <FormControl isRequired>
              <FormLabel>Base URL</FormLabel>
              <Input
                value={formData.base_url}
                onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                placeholder="https://api.example.com"
                type="url"
              />
              <FormHelperText>The base URL/endpoint for the database API</FormHelperText>
            </FormControl>

            <FormControl>
              <FormLabel>API Key</FormLabel>
              <Input
                value={formData.api_key}
                onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                placeholder="Your API key (if required)"
                type="password"
              />
              <FormHelperText>Leave blank to keep the stored key. The value is never shown after save.</FormHelperText>
            </FormControl>

            <FormControl>
              <FormLabel>Vault secret name (optional)</FormLabel>
              <Input
                value={apiKeySecretName}
                onChange={(e) => setApiKeySecretName(e.target.value.toUpperCase())}
                placeholder="MY_DB_API_KEY"
              />
              <FormHelperText>Use a secret from Settings → Secrets instead of pasting a raw key.</FormHelperText>
            </FormControl>

            <FormControl>
              <FormLabel>Adapter Type</FormLabel>
              <Select
                value={formData.adapter_type}
                onChange={(e) => setFormData({ ...formData, adapter_type: e.target.value })}
              >
                <option value="rest_api">REST API</option>
                <option value="graphql">GraphQL (Coming Soon)</option>
                <option value="sdk">SDK (Coming Soon)</option>
              </Select>
              <FormHelperText>The type of API adapter to use</FormHelperText>
            </FormControl>

            <FormControl>
              <Checkbox
                isChecked={formData.is_active}
                onChange={(e) => setFormData({ ...formData, is_active: e.target.checked })}
              >
                Active (Enable this database for parameter suggestions)
              </Checkbox>
            </FormControl>

            <Divider />

            <Box>
              <Button
                onClick={() => setShowAdvanced(!showAdvanced)}
                variant="ghost"
                size="sm"
                rightIcon={showAdvanced ? <ChevronUpIcon /> : <ChevronDownIcon />}
              >
                Advanced Settings
              </Button>
              
              <Collapse in={showAdvanced} animateOpacity>
                <VStack spacing={4} align="stretch" mt={4} p={4} bg="gray.50" borderRadius="md">
                  <FormControl>
                    <FormLabel>Query Endpoint</FormLabel>
                    <Input
                      value={queryEndpoint}
                      onChange={(e) => setQueryEndpoint(e.target.value)}
                      placeholder="/query or /search or /api/query"
                    />
                    <FormHelperText>
                      API endpoint path for querying (e.g., /query, /search, /datasets). 
                      Leave empty to use default patterns.
                    </FormHelperText>
                  </FormControl>

                  <FormControl>
                    <FormLabel>Authentication Type</FormLabel>
                    <Select
                      value={authType}
                      onChange={(e) => setAuthType(e.target.value)}
                    >
                      <option value="none">None</option>
                      <option value="api_key">API Key</option>
                      <option value="bearer">Bearer Token</option>
                      <option value="basic">Basic Auth</option>
                    </Select>
                    <FormHelperText>How to authenticate with the API</FormHelperText>
                  </FormControl>

                  {authType === 'api_key' && (
                    <FormControl>
                      <FormLabel>API Key Header Name</FormLabel>
                      <Input
                        value={apiKeyHeader}
                        onChange={(e) => setApiKeyHeader(e.target.value)}
                        placeholder="X-API-Key"
                      />
                      <FormHelperText>
                        HTTP header name for API key (e.g., X-API-Key, Authorization)
                      </FormHelperText>
                    </FormControl>
                  )}

                  <FormControl>
                    <FormLabel>Custom Configuration (JSON)</FormLabel>
                    <Textarea
                      value={customConfigJson}
                      onChange={(e) => setCustomConfigJson(e.target.value)}
                      placeholder='{"query_params_template": {}, "timeout": 10, ...}'
                      fontFamily="mono"
                      fontSize="sm"
                      rows={6}
                    />
                    <FormHelperText>
                      Additional configuration as JSON (optional). 
                      Will be merged with the settings above.
                    </FormHelperText>
                  </FormControl>
                </VStack>
              </Collapse>
            </Box>

            <Divider />

            <HStack>
              <Button
                onClick={handleTestConnection}
                isLoading={testing}
                loadingText="Testing..."
                colorScheme="blue"
                variant="outline"
                flex={1}
              >
                Test Connection
              </Button>
            </HStack>

            {testResult && (
              <Alert
                status={testResult.success ? 'success' : 'error'}
                borderRadius="md"
              >
                <AlertIcon />
                <Box flex="1">
                  <AlertTitle>
                    {testResult.success ? 'Connection Successful' : 'Connection Failed'}
                  </AlertTitle>
                  <AlertDescription>
                    {testResult.message}
                    {testResult.suggestions && testResult.suggestions.length > 0 && (
                      <VStack align="stretch" mt={2} spacing={1}>
                        {testResult.suggestions.map((suggestion: string, idx: number) => (
                          <Text key={idx} fontSize="sm">
                            {suggestion}
                          </Text>
                        ))}
                      </VStack>
                    )}
                  </AlertDescription>
                </Box>
              </Alert>
            )}

            {isEditMode && database && (
              <>
                <Divider />
                <VStack align="stretch" spacing={2}>
                  <Text fontWeight="bold">Status</Text>
                  <HStack>
                    <Badge colorScheme={database.is_verified ? 'green' : 'red'}>
                      {database.is_verified ? (
                        <>
                          <CheckIcon mr={1} /> Verified
                        </>
                      ) : (
                        <>
                          <CloseIcon mr={1} /> Not Verified
                        </>
                      )}
                    </Badge>
                    <Badge colorScheme={database.is_active ? 'green' : 'gray'}>
                      {database.is_active ? 'Active' : 'Inactive'}
                    </Badge>
                  </HStack>
                  {database.last_tested && (
                    <Text fontSize="sm" color="gray.500">
                      Last tested: {new Date(database.last_tested).toLocaleString()}
                    </Text>
                  )}
                </VStack>
              </>
            )}
          </VStack>
        </ModalBody>

        <ModalFooter>
          <HStack spacing={2}>
            {isEditMode && (
              <Button
                onClick={handleDelete}
                colorScheme="red"
                variant="ghost"
              >
                Delete
              </Button>
            )}
            <Button onClick={onClose} variant="ghost">
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              isLoading={saving}
              loadingText="Saving..."
              colorScheme="blue"
            >
              {isEditMode ? 'Update' : 'Create'}
            </Button>
          </HStack>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default CustomDatabaseModal;
