import React, { useState, useEffect } from 'react';
import {
  VStack,
  HStack,
  Box,
  Text,
  Code,
  SimpleGrid,
  Flex,
  Button,
  Tooltip,
  Input,
  Textarea,
  IconButton,
  useToast,
} from '@chakra-ui/react';
import { EditIcon, CheckIcon, CloseIcon, ViewIcon } from '@chakra-ui/icons';
import { CalculationNodeData, SchemaFields } from '../type';
import { Node } from '@xyflow/react';
import { createAuthHeaders } from '../../../api/authHeaders';
import HomeView from '../homeView';
import convertToStrIncFloat from '/homeView';
 
interface NodeDetailsContentProps {
  nodeData: Node<CalculationNodeData> | null;
  onNodeUpdate?: (nodeId: string, updatedData: Partial<CalculationNodeData>) => void;
  onRefreshNodeData?: (filename: string) => Promise<any>;
  onViewCode?: () => void;
  workflowId?: string;
  convertToStrIncFloat: (obj: any) => any; 
}

// Jupyterを別タブで開く
const OpenJupyter = (filename : string, category : string) => {
    window.open("http://localhost:8000/user/user1/lab/workspaces/auto-E/tree/codes/nodes/"+category.toLowerCase()+"/"+filename+".py", "_blank");
};

const NodeDetailsContent: React.FC<NodeDetailsContentProps> = ({ nodeData, onNodeUpdate, onRefreshNodeData, onViewCode, workflowId, convertToStrIncFloat }) => {
  const [editingInstance, setEditingInstance] = useState<string>('');
  const [editingParam, setEditingParam] = useState<string | null>(null);
  const [editingField, setEditingField] = useState<'default_value' | 'constraints' | null>(null);
  const [editValue, setEditValue] = useState<string>('');
  const [localNodeData, setLocalNodeData] = useState<Node<CalculationNodeData> | null>(nodeData);
  const toast = useToast();

  // nodeDataが変更されたときにローカル状態を更新し、編集状態をリセット
  useEffect(() => {
    console.log('NodeDetailsContent: nodeData changed', nodeData);
    setLocalNodeData(nodeData);
    
    // 編集状態をリセット（パラメーター更新後に古い編集状態が残らないように）
    setEditingInstance('');
    setEditingParam(null);
    setEditingField(null);
    setEditValue('');
  }, [nodeData]);

  // インスタンス名の更新API呼び出し
  const updateInstanceName = async (instanceName: string) => {
    try {
      console.log('=== Parameter Update Debug Info ===');
      console.log('Node ID:', localNodeData?.id);
      console.log('Workflow ID:', workflowId);
      console.log('Innstance Name:', instanceName);

      let response;
      let requestBody;

      // ワークフロー内のノード - ワークフローパラメーター更新エンドポイントを使用
      const endpoint = `/api/workflow/${workflowId}/nodes/${localNodeData.id}/instance_name/`;
      console.log('Using workflow instance_name endpoint:', endpoint);

      requestBody = {
        instance_name: instanceName
      };
      console.log('Request body for workflow node:', JSON.stringify(requestBody, null, 2));

      response = await fetch(endpoint, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      console.log('Response status:', response.status);
      console.log('Response URL:', response.url);

      if (!response.ok) {
        const responseText = await response.text();
        console.error('Error response body:', responseText);
        throw new Error(`HTTP error! status: ${response.status}, body: ${responseText}`);
      }

      // 成功レスポンスのボディもログに出力
      const responseText = await response.text();
      console.log('Success response body:', responseText);

      // DBから最新データを再取得またはローカル状態を更新
      if (localNodeData && onNodeUpdate) {
        console.log('Starting post-update refresh process for node:', localNodeData.id);
        console.log('onRefreshNodeData available:', !!onRefreshNodeData);

        let updatedInstanceName: string | undefined;

        // ワークフロー内のノード - instanceNameを直接更新
        updatedInstanceName = localNodeData.data.instanceName;

        console.log('Updating workflow node instance name:', {
          nodeId: localNodeData.id,
          instanceName
        });

        // ローカル状態を即座に更新
        const updatedNodeData = {
          ...localNodeData,
          data: {
            ...localNodeData.data,
            instanceName: updatedInstanceName,
            __timestamp: Date.now()
          }
        };
        setLocalNodeData(updatedNodeData);

        // 親コンポーネントの状態も更新
        onNodeUpdate(localNodeData.id, {
          instanceName: updatedInstanceName,
          __timestamp: Date.now()
        });
      }

      toast({
        title: "Success",
        description: `Instance name updated successfully`,
        status: "success",
        duration: 3000,
        isClosable: true,
      });

      return true;
    } catch (error) {
      console.error('Error updating instance_name:', error);
      toast({
        title: "Error",
        description: `Failed to update instance_name: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return false;
    }
  };

  // パラメータの更新API呼び出し
  const updateParameter = async (parameterKey: string, parameterValue: any, parameterField: 'default_value' | 'constraints') => {
    try {
      // ワークフロー内のノードかどうかを判定
      const isWorkflowNode = localNodeData && !localNodeData.id.startsWith('sidebar_');

      console.log('=== Parameter Update Debug Info ===');
      console.log('Node ID:', localNodeData?.id);
      console.log('Workflow ID:', workflowId);
      console.log('Is Workflow Node:', isWorkflowNode);
      console.log('Parameter Key:', parameterKey);
      console.log('Parameter Value:', parameterValue);
      console.log('Parameter Field:', parameterField);

      let response;
      let requestBody;

      if (isWorkflowNode) {
        // ワークフロー内のノード - ワークフローパラメーター更新エンドポイントを使用
        const endpoint = `/api/workflow/${workflowId}/nodes/${localNodeData.id}/parameters/`;
        console.log('Using workflow parameters endpoint:', endpoint);

        requestBody = {
          parameter_key: parameterKey,
          parameter_field: parameterField,
          parameter_value: parameterValue
        };
        console.log('Request body for workflow node:', JSON.stringify(requestBody, null, 2));

        response = await fetch(endpoint, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      } else {
        // サイドバーのノード - 既存のエンドポイントを使用
        const endpoint = '/api/box/parameters/update/';
        console.log('Using sidebar endpoint:', endpoint);

        requestBody = {
          parameter_key: parameterKey,
          parameter_field: parameterField,
          parameter_value: parameterValue,
          filename: localNodeData.data.file_name
        };
        console.log('Request body for sidebar node:', JSON.stringify(requestBody, null, 2));

        response = await fetch(endpoint, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      }

      console.log('Response status:', response.status);
      console.log('Response URL:', response.url);

      if (!response.ok) {
        const responseText = await response.text();
        console.error('Error response body:', responseText);
        throw new Error(`HTTP error! status: ${response.status}, body: ${responseText}`);
      }

      // 成功レスポンスのボディもログに出力
      const responseText = await response.text();
      console.log('Success response body:', responseText);

      // レスポンスが空でなければJSONとしてパース
      let responseData = null;
      if (responseText.trim()) {
        try {
          responseData = JSON.parse(responseText);
          console.log('Parsed response data:', responseData);
        } catch (e) {
          console.log('Response is not valid JSON, treating as plain text');
        }
      }

      // DBから最新データを再取得またはローカル状態を更新
      if (localNodeData && onNodeUpdate) {
        console.log('Starting post-update refresh process for node:', localNodeData.id);
        console.log('onRefreshNodeData available:', !!onRefreshNodeData);

        let updatedSchema: SchemaFields | undefined;

        if (isWorkflowNode) {
          // ワークフロー内のノード - schemaを直接更新
          updatedSchema = { ...localNodeData.data.schema };
          if (updatedSchema.parameters && updatedSchema.parameters[parameterKey]) {
            updatedSchema.parameters[parameterKey] = {
              ...updatedSchema.parameters[parameterKey],
              [parameterField]: parameterValue
            };

            console.log('Updating workflow node schema:', {
              nodeId: localNodeData.id,
              parameterKey,
              parameterField,
              parameterValue
            });

            // ローカル状態を即座に更新
            const updatedNodeData = {
              ...localNodeData,
              data: {
                ...localNodeData.data,
                schema: updatedSchema,
                __timestamp: Date.now()
              }
            };
            setLocalNodeData(updatedNodeData);

            // 親コンポーネントの状態も更新
            onNodeUpdate(localNodeData.id, {
              schema: updatedSchema,
              __timestamp: Date.now()
            });
          }
        } else {
          // サイドバーのノード - スキーマを更新
          // まずローカル状態を即座に更新（即時反映のため）
          updatedSchema = { ...localNodeData.data.schema };
          if (updatedSchema.parameters && updatedSchema.parameters[parameterKey]) {
            updatedSchema.parameters[parameterKey] = {
              ...updatedSchema.parameters[parameterKey],
              [parameterField]: parameterValue
            };

            console.log('Immediately updating sidebar node with new parameter value:', {
              nodeId: localNodeData.id,
              parameterKey,
              parameterField,
              parameterValue
            });

            // ローカル状態を即座に更新（モーダル内の表示も更新される）
            const updatedNodeData = {
              ...localNodeData,
              data: {
                ...localNodeData.data,
                schema: updatedSchema,
                __timestamp: Date.now()
              }
            };
            setLocalNodeData(updatedNodeData);

            // 親コンポーネントの状態も更新
            onNodeUpdate(localNodeData.id, {
              schema: updatedSchema,
              __timestamp: Date.now()
            });
          }
        }

        // サイドバーノードの場合のみサーバーから最新データを取得（データ整合性のため）
        if (onRefreshNodeData && !isWorkflowNode) {
          try {
            console.log('Attempting to refresh data for sidebar node file:', localNodeData.data.file_name);
            const refreshedData = await onRefreshNodeData(localNodeData.data.file_name);
            console.log('Refresh result:', refreshedData);

            if (refreshedData && refreshedData.schema) {
              console.log('Updating sidebar node with refreshed schema from server:', refreshedData.schema);
              updatedSchema = refreshedData.schema;

              // ローカル状態も更新
              const finalUpdatedNodeData = {
                ...localNodeData,
                data: {
                  ...localNodeData.data,
                  schema: updatedSchema,
                  __timestamp: Date.now()
                }
              };
              setLocalNodeData(finalUpdatedNodeData);

              // 親コンポーネントの状態も更新
              onNodeUpdate(localNodeData.id, {
                schema: updatedSchema,
                __timestamp: Date.now()
              });
            }
          } catch (error) {
            console.error('Failed to refresh sidebar node data from server:', error);
            // サーバーからの取得に失敗してもローカル更新は既に済んでいる
          }
        }
        
        // 同期処理は削除 - サイドバーとワークフローノードは独立して扱う
      }

      toast({
        title: "Success",
        description: `Parameter ${parameterField} updated successfully`,
        status: "success",
        duration: 3000,
        isClosable: true,
      });

      return true;
    } catch (error) {
      console.error('Error updating parameter:', error);
      toast({
        title: "Error",
        description: `Failed to update parameter: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return false;
    }
  };

  // 編集開始 (インスタンス名)
  const startInstanceEditing = (currentValue: string) => {
    setEditingInstance(currentValue);
  };

  // 編集保存 (インスタンス名)
  const saveInstanceEdit = async () => {
    const success = await updateInstanceName(editingInstance);
    if (success) {
      // 編集状態をクリア
      setEditingInstance('');
    }
  };

  // 編集キャンセル (インスタンス名)
  const cancelInstanceEdit = () => {
    setEditingInstance('');
  };

  // 編集開始
  const startEditing = (paramKey: string, field: 'default_value' | 'constraints', currentValue: any) => {
    setEditingParam(paramKey);
    setEditingField(field);
    
    // 配列や複雑なオブジェクトを適切にフォーマット
    if (Array.isArray(currentValue)) {
      setEditValue(JSON.stringify(currentValue, null, 2));
    } else if (typeof currentValue === 'object' && currentValue !== null) {
      setEditValue(JSON.stringify(currentValue, null, 2));
    } else if (typeof currentValue === 'string') {
      setEditValue(currentValue);
    } else {
      setEditValue(JSON.stringify(currentValue));
    }
  };

  // 編集保存
  const saveEdit = async () => {
    if (!editingParam || !editingField) return;

    let parsedValue: any;
    try {
      // まずJSONとして解析を試行
      parsedValue = JSON.parse(editValue);
      
      // 配列の場合の検証
      if (Array.isArray(parsedValue)) {
        console.log('Parsed array value:', parsedValue);
      }
    } catch (error) {
      // JSON解析に失敗した場合、文字列として扱う
      console.log('JSON parse failed, treating as string:', editValue);
      parsedValue = editValue;
    }

    const success = await updateParameter(editingParam, parsedValue, editingField);
    if (success) {
      // 編集状態をクリア
      setEditingParam(null);
      setEditingField(null);
      setEditValue('');
    }
  };

  // 編集キャンセル
  const cancelEdit = () => {
    setEditingParam(null);
    setEditingField(null);
    setEditValue('');
  };

  if (!localNodeData) {
    return (
      <Flex align="center" justify="center" h="200px">
        <Text color="gray.500" fontStyle="italic" fontSize="lg">
          No node selected
        </Text>
      </Flex>
    );
  }

  console.log("これデータ", localNodeData);
  console.log("NodeData timestamp in modal:", localNodeData.data.__timestamp || 'no timestamp');

  const schema: SchemaFields = localNodeData.data.schema || { inputs: {}, outputs: {}, parameters: {}, methods: {} };
  
  const renderDataTypeColor = (type: string) => {
    const colorMap: Record<string, string> = {
      'OBJECT': 'purple',
      'DICT': 'blue',
      'BOOL': 'green',
      'BOOLEAN': 'green',
      'INT': 'orange',
      'FLOAT': 'teal',
      'STR': 'pink',
      'STRING': 'pink',
      'LIST': 'cyan',
      'ARRAY': 'cyan',
      'ANY': 'gray',
    };
    return colorMap[type.toUpperCase()] || 'gray';
  };

  // ノード固有のパラメーター値を取得するヘルパー関数
  const getNodeParameterValue = (parameterKey: string, field: 'default_value' | 'constraints'): any => {
    // 全てのノードでschemaから最新の値を取得（DBの最新状態を反映）
    const param = localNodeData?.data.schema.parameters?.[parameterKey];
    if (param && param[field] !== undefined) {
      //if (field == 'default_value') {
      //  return Number.isInteger(param[field]) ? `${param[field].toFixed(1)}` : `${param[field]}`;
      //}
      return param[field];
    }

    return undefined;
  };

  // データをクリーンに表示するためのヘルパー関数
  const formatDataForDisplay = (data: any) => {
    if (Array.isArray(data)) {
      // 配列の場合、各要素を簡潔に表示
      if (data.length === 0) {
        return '[]';
      }
      
      // 配列の長さが5を超える場合は省略表示
      if (data.length > 5) {
        const firstFew = data.slice(0, 3).map(item => {
          if (typeof item === 'object' && item !== null) {
            return JSON.stringify(item);
          }
          return String(item);
        });
        return `[${firstFew.join(', ')}, ...${data.length - 3} more]`;
      }
      
      return `[${data.map(item => {
        if (typeof item === 'object' && item !== null) {
          // オブジェクトの場合、キーのみまたは重要なプロパティのみを表示
          if (item.name) return `"${item.name}"`;
          if (item.type) return `"${item.type}"`;
          return JSON.stringify(item);
        }
        return typeof item === 'string' ? `"${item}"` : String(item);
      }).join(', ')}]`;
    }
    
    if (typeof data === 'object' && data !== null) {
      return JSON.stringify(data);
    }
    
    return String(data);
  };

  const renderInstanceNameSection = () => {    
    return (
      <HStack align="start" spacing={2}>
        {editingInstance != '' ? (
          <VStack flex="1" spacing={1} align="stretch">                      
            <Input
              value={editingInstance }
              onChange={(e) => setEditingInstance(e.target.value)}
              size="xs"
              bg="gray.600"
              color="white"
              fontSize="xl"
              placeholder="Instance name"
            />
            <HStack spacing={1}>
              <IconButton
                aria-label="Save"
                icon={<CheckIcon />}
                size="xs"
                colorScheme="green"
                onClick={saveInstanceEdit}
              />
              <IconButton
                aria-label="Cancel"
                icon={<CloseIcon />}
                size="xs"
                colorScheme="red"
                onClick={cancelInstanceEdit}
              />
            </HStack>
          </VStack>
        ) : (
          <HStack flex="1" spacing={1}>
            <Text fontWeight="bold" fontSize="xl" color="purple.300">
              {localNodeData.data.instanceName || localNodeData.data.label }
            </Text>
            <Tooltip label="Edit instance_name" hasArrow>
              <IconButton
                aria-label="Edit instance_name"
                icon={<EditIcon />}
                size="xs"
                colorScheme="blue"
                variant="ghost"
                onClick={() => startInstanceEditing(localNodeData.data.instanceName || '')}
              />
            </Tooltip>
          </HStack>
        )}
      </HStack>
    );
  };

  const renderParametersSection = () => {
    if (!schema.parameters || Object.keys(schema.parameters).length === 0) {
      return (
        <Flex align="center" justify="center" h="100px">
          <Text color="gray.500" fontStyle="italic">
            No parameters defined
          </Text>
        </Flex>
      );
    }

    return (
      <VStack spacing={3} align="stretch">
        <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6} w="100%" templateColumns={{ lg: "1fr 1fr" }}>
        {Object.entries(schema.parameters).map(([key, param]) => (
          <Box 
            key={key} 
            p={4} 
            bg="gray.700" 
            borderRadius="md" 
            borderWidth="1px" 
            borderColor="orange.400"
            boxShadow="sm"
          >
            <VStack align="stretch" spacing={3}>
              <Text fontWeight="bold" fontSize="md" color="orange.200">"{key}"</Text>
              
              {param.description && (
                <HStack align="start">
                  <Text fontSize="xs" color="gray.400" minW="80px">description:</Text>
                  <Text fontSize="sm" color="gray.300">
                    {param.description}
                  </Text>
                </HStack>
              )}
              
              <VStack align="stretch" spacing={2}>
                {/* Default Value - 編集可能 */}
                {(param.default_value !== undefined || getNodeParameterValue(key, 'default_value') !== undefined) && (
                  <HStack align="start" spacing={2}>
                    <Text fontSize="xs" color="gray.400" minW="80px">default_value:</Text>
                    {editingParam === key && editingField === 'default_value' ? (
                      <VStack flex="1" spacing={1} /*align="stretch"*/>
                        {editValue.includes('\n') || editValue.startsWith('[') || editValue.startsWith('{') ? (
                          <Textarea
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            size="xs"
                            bg="gray.600"
                            color="white"
                            fontSize="xs"
                            minH="80px"
                            resize="vertical"
                            placeholder="配列の場合: [1, 2, 3] または ['a', 'b', 'c']"
                          />
                        ) : (
                          <Input
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            size="xs"
                            bg="gray.600"
                            color="white"
                            fontSize="xs"
                          />
                        )}
                        <HStack spacing={1}>
                          <IconButton
                            aria-label="Save"
                            icon={<CheckIcon />}
                            size="xs"
                            colorScheme="green"
                            onClick={saveEdit}
                          />
                          <IconButton
                            aria-label="Cancel"
                            icon={<CloseIcon />}
                            size="xs"
                            colorScheme="red"
                            onClick={cancelEdit}
                          />
                        </HStack>
                      </VStack>
                    ) : (
                      <HStack flex="1" spacing={1}>
                        <Code colorScheme="gray" fontSize="xs" bg="gray.600" color="white" flex="1" maxW="360">
                          {(() => {
                            var currentValue = getNodeParameterValue(key, 'default_value');
                            //return convertToStrIncFloat(currentValue);
                            return Array.isArray(currentValue) || typeof currentValue === 'object'
                              ? formatDataForDisplay(currentValue)
                              : typeof currentValue === 'string'
                                ? `"${currentValue}"`
                                : String(currentValue);
                          })()}
                        </Code>
                        <Tooltip label="Edit default value" hasArrow>
                          <IconButton
                            aria-label="Edit default value"
                            icon={<EditIcon />}
                            size="xs"
                            colorScheme="blue"
                            variant="ghost"
                            onClick={() => startEditing(key, 'default_value', getNodeParameterValue(key, 'default_value'))}
                          />
                        </Tooltip>
                      </HStack>
                    )}
                  </HStack>
                )}
                
                {/* Constraints - 編集可能 */}
                <HStack align="start" spacing={2}>
                  <Text fontSize="xs" color="gray.400" minW="80px">constraints:</Text>
                  {editingParam === key && editingField === 'constraints' ? (
                    <VStack flex="1" spacing={1} align="stretch">
                      {editValue.includes('\n') || editValue.startsWith('[') || editValue.startsWith('{') ? (
                        <Textarea
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          size="xs"
                          bg="gray.600"
                          color="white"
                          fontSize="xs"
                          minH="80px"
                          resize="vertical"
                          placeholder="制約条件 (JSON形式): {'min': 0, 'max': 100} または ['option1', 'option2']"
                        />
                      ) : (
                        <Input
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          size="xs"
                          bg="gray.600"
                          color="white"
                          fontSize="xs"
                          placeholder="制約条件 (JSON形式)"
                        />
                      )}
                      <HStack spacing={1}>
                        <IconButton
                          aria-label="Save"
                          icon={<CheckIcon />}
                          size="xs"
                          colorScheme="green"
                          onClick={saveEdit}
                        />
                        <IconButton
                          aria-label="Cancel"
                          icon={<CloseIcon />}
                          size="xs"
                          colorScheme="red"
                          onClick={cancelEdit}
                        />
                      </HStack>
                    </VStack>
                  ) : (
                    <HStack flex="1" spacing={1}>
                      <Code colorScheme="blue" fontSize="xs" bg="blue.600" color="white" flex="1">
                        {(() => {
                          const currentConstraints = getNodeParameterValue(key, 'constraints');
                          return currentConstraints ? formatDataForDisplay(convertToStrIncFloat(currentConstraints)) : 'None';
                          //return currentConstraints ? formatDataForDisplay(currentConstraints) : 'None';
                        })()}
                      </Code>
                      <Tooltip label="Edit constraints" hasArrow>
                        <IconButton
                          aria-label="Edit constraints"
                          icon={<EditIcon />}
                          size="xs"
                          colorScheme="blue"
                          variant="ghost"
                          onClick={() => startEditing(key, 'constraints', getNodeParameterValue(key, 'constraints') || '')}
                        />
                      </Tooltip>
                    </HStack>
                  )}
                </HStack>
              </VStack>
            </VStack>
          </Box>
        ))}
        </SimpleGrid>
      </VStack>
    );
  };

  const renderPortsSection = (ports: Record<string, any>, title: string, colorScheme: string) => {
    if (!ports || Object.keys(ports).length === 0) {
      return (
        <Flex align="center" justify="center" h="100px">
          <Text color="gray.500" fontStyle="italic">
            No {title.toLowerCase()} defined
          </Text>
        </Flex>
      );
    }

    return (
      <VStack spacing={3} align="stretch">
        {Object.entries(ports).map(([portName, portData]) => (
          <HStack key={portName} spacing={3} align="center">
            <Text fontWeight="bold" fontSize="md" color={`${colorScheme}.200`}>
              {portName}{portData.optional ? '' : '*'}
            </Text>
            <Text color="gray.400">:</Text>
            <Text 
              fontWeight="semibold" 
              fontSize="md" 
              color={`${renderDataTypeColor(portData.type || 'any')}.300`}
            >
              type={portData.type || 'any'}<br/>
              description={portData.description || 'any'}<br/>
            </Text>
          </HStack>
        ))}
      </VStack>
    );
  };

  const renderMethodsSection = () => {
    if (!schema.methods || Object.keys(schema.methods).length === 0) {
      return (
        <Flex align="center" justify="center" h="100px">
          <Text color="gray.500" fontStyle="italic">
            No methods defined
          </Text>
        </Flex>
      );
    }

    return (
      <VStack spacing={3} align="stretch">
        {Object.entries(schema.methods).map(([methodName, method]) => (
          <Box 
            key={methodName} 
            p={4} 
            bg="gray.700" 
            borderRadius="md" 
            borderWidth="1px" 
            borderColor="purple.400"
            boxShadow="sm"
          >
            <VStack align="stretch" spacing={3}>
              <Text fontWeight="bold" fontSize="md" color="purple.200">"{methodName}"</Text>
              
              {method.description && (
                <HStack align="start">
                  <Text fontSize="xs" color="gray.400" minW="80px">description:</Text>
                  <Text fontSize="sm" color="gray.300">
                    {method.description}
                  </Text>
                </HStack>
              )}
              
              <VStack align="stretch" spacing={2}>
                {method.inputs && method.inputs.length > 0 && (
                  <HStack align="start">
                    <Text fontSize="xs" color="gray.400" minW="80px">inputs:</Text>
                    <Code colorScheme="blue" fontSize="xs" bg="blue.600" color="white">
                      {formatDataForDisplay(method.inputs)}
                    </Code>
                  </HStack>
                )}
                
                {method.outputs && method.outputs.length > 0 && (
                  <HStack align="start">
                    <Text fontSize="xs" color="gray.400" minW="80px">outputs:</Text>
                    <Code colorScheme="green" fontSize="xs" bg="green.600" color="white">
                      {formatDataForDisplay(method.outputs)}
                    </Code>
                  </HStack>
                )}
              </VStack>
            </VStack>
          </Box>
        ))}
      </VStack>
    );
  };


  return (
    <>
      <Box p={6} h="100%" overflowY="auto" maxW="none" w="100%">
        <VStack spacing={6} align="stretch" maxW="none">
          {/* Node Info Header */}
          <Box bg="gray.800" borderRadius="lg" boxShadow="md" marginTop={-5} p={4}>
            <Flex justify="space-between" align="start">
              <VStack align="start" spacing={1}>
                {renderInstanceNameSection()}
                <Text fontSize="sm" color="gray.400">
                  Node ID: {localNodeData.id}
                </Text>
                {localNodeData.data.__timestamp && (
                  <Text fontSize="xs" color="gray.500">
                    Last updated: {new Date(localNodeData.data.__timestamp).toLocaleTimeString()}
                  </Text>
                )}
              </VStack>
            </Flex>
          </Box>

          {/* 4つのセクションを2x2グリッドで配置 */}
          <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6} w="100%" templateColumns={{ lg: "1fr 1fr" }}>
            <Box>
              {/* Inputs */}
              <Box>
                <Text fontWeight="bold" fontSize="lg" mb={4} color="blue.300">
                  ・Inputs
                </Text>
                <Box
                  bg="gray.800"
                  p={6}
                  borderRadius="lg"
                  border="2px"
                  borderColor="blue.500"
                  h="220px"
                  overflowY="auto"
                  boxShadow="lg"
                >
                  {renderPortsSection(schema.inputs || {}, 'Inputs', 'blue')}
                </Box>
              </Box>

              {/* Outputs */}
              <Box marginTop={4}>
                <Text fontWeight="bold" fontSize="lg" mb={4} color="green.300">
                  ・Outputs
                </Text>
                <Box
                  bg="gray.800"
                  p={6}
                  borderRadius="lg"
                  border="2px"
                  borderColor="green.500"
                  h="220px"
                  overflowY="auto"
                  boxShadow="lg"
                >
                  {renderPortsSection(schema.outputs || {}, 'Outputs', 'green')}
                </Box>
              </Box>
            </Box>
            {/* Methods */}
            <Box>
              <Text fontWeight="bold" fontSize="lg" mb={4} color="purple.300">
                ・Methods
              </Text>
              <Box
                bg="gray.800"
                p={6}
                borderRadius="lg"
                border="2px"
                borderColor="purple.500"
                h="500px"
                overflowY="auto"
                boxShadow="lg"
              >
                {renderMethodsSection()}
              </Box>
            </Box>
          </SimpleGrid>
          {/* Parameters */}
          <Box>
            <Text fontWeight="bold" fontSize="lg" mb={4} color="orange.300">
              ・Parameters
            </Text>
            <Box
              bg="gray.800"
              p={6}
              borderRadius="lg"
              border="2px"
              borderColor="orange.500"
              h="500px"
              overflowY="auto"
              boxShadow="lg"
            >
              {renderParametersSection()}
            </Box>
          </Box>
        </VStack>
      </Box>

    </>
  );
};

export default NodeDetailsContent;
