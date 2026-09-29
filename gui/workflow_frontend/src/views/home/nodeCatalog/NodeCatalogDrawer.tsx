import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  AlertIcon,
  Badge,
  Box,
  Drawer,
  DrawerBody,
  DrawerCloseButton,
  DrawerContent,
  DrawerHeader,
  DrawerOverlay,
  Flex,
  Heading,
  HStack,
  Input,
  InputGroup,
  InputLeftElement,
  Select,
  Spinner,
  Text,
  useColorModeValue,
  VStack,
} from "@chakra-ui/react";
import { SearchIcon } from "@chakra-ui/icons";
import { useUploadedNodes } from "../../../hooks/useUploadedNodes";
import { useNodeCatalog } from "./NodeCatalogContext";
import {
  filterNodeCatalog,
  formatParamDefault,
  groupNodesByCategory,
  isPortRequired,
  nodeCatalogKey,
  uniqueCatalogCategories,
  type CatalogNode,
} from "./nodeCatalogSearch";
import type { InputField, OutputField, ParameterField } from "../type";

const SectionLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const muted = useColorModeValue("gray.500", "gray.400");
  return (
    <Text
      fontSize="xs"
      textTransform="uppercase"
      letterSpacing="0.04em"
      color={muted}
      mb={2}
      fontWeight="semibold"
    >
      {children}
    </Text>
  );
};

const PortRow: React.FC<{
  name: string;
  field: InputField | OutputField;
}> = ({ name, field }) => {
  const muted = useColorModeValue("gray.500", "gray.400");
  const required = isPortRequired(field);
  return (
    <Box py={2} borderBottomWidth="1px" borderColor={useColorModeValue("#eee", "gray.700")}>
      <HStack spacing={2} align="center" flexWrap="wrap">
        <Text fontWeight="semibold" fontSize="sm">
          {name}
        </Text>
        {field.type && (
          <Badge fontFamily="mono" fontSize="xs" variant="subtle" colorScheme="blue">
            {field.type}
          </Badge>
        )}
        {required && (
          <Badge fontSize="xs" colorScheme="orange">
            required
          </Badge>
        )}
      </HStack>
      {field.description ? (
        <Text fontSize="sm" color={muted} mt={1}>
          {field.description}
        </Text>
      ) : null}
    </Box>
  );
};

const ParamRow: React.FC<{ name: string; field: ParameterField }> = ({
  name,
  field,
}) => {
  const muted = useColorModeValue("gray.500", "gray.400");
  const defaultText = formatParamDefault(field.default_value);
  return (
    <Box py={2} borderBottomWidth="1px" borderColor={useColorModeValue("#eee", "gray.700")}>
      <HStack spacing={2} align="center" flexWrap="wrap">
        <Text fontWeight="semibold" fontSize="sm">
          {name}
        </Text>
        {field.type && (
          <Badge fontFamily="mono" fontSize="xs" variant="subtle" colorScheme="purple">
            {field.type}
          </Badge>
        )}
      </HStack>
      {field.description ? (
        <Text fontSize="sm" color={muted} mt={1}>
          {field.description}
        </Text>
      ) : null}
      {defaultText !== null && (
        <Text fontSize="xs" color={muted} mt={1} fontFamily="mono">
          default: {defaultText}
        </Text>
      )}
    </Box>
  );
};

const NodeDetail: React.FC<{ node: CatalogNode }> = ({ node }) => {
  const muted = useColorModeValue("gray.500", "gray.400");
  const schemaMissing = node.schema == null;
  const inputs = Object.entries(node.schema?.inputs ?? {});
  const outputs = Object.entries(node.schema?.outputs ?? {});
  const parameters = Object.entries(node.schema?.parameters ?? {});

  return (
    <VStack align="stretch" spacing={5}>
      <Box>
        <Heading size="sm">{node.label}</Heading>
        <HStack mt={2} spacing={2} flexWrap="wrap">
          {node.category && <Badge colorScheme="blue">{node.category}</Badge>}
          {node.class_name && node.class_name !== node.label && (
            <Badge variant="outline">{node.class_name}</Badge>
          )}
        </HStack>
        {node.file_name && (
          <Text fontSize="xs" color={muted} mt={2} fontFamily="mono">
            {node.file_name}
          </Text>
        )}
      </Box>
      {node.description ? (
        <Box>
          <SectionLabel>Description</SectionLabel>
          <Text whiteSpace="pre-wrap">{node.description}</Text>
        </Box>
      ) : null}
      {schemaMissing ? (
        <Text fontSize="sm" color={muted}>
          Schema was not parsed for this file.
        </Text>
      ) : (
        <>
          <Box>
            <SectionLabel>Inputs ({inputs.length})</SectionLabel>
            {inputs.length === 0 ? (
              <Text fontSize="sm" color={muted}>
                None
              </Text>
            ) : (
              inputs.map(([name, field]) => (
                <PortRow key={`in-${name}`} name={name} field={field} />
              ))
            )}
          </Box>
          <Box>
            <SectionLabel>Outputs ({outputs.length})</SectionLabel>
            {outputs.length === 0 ? (
              <Text fontSize="sm" color={muted}>
                None
              </Text>
            ) : (
              outputs.map(([name, field]) => (
                <PortRow key={`out-${name}`} name={name} field={field} />
              ))
            )}
          </Box>
          <Box>
            <SectionLabel>Parameters ({parameters.length})</SectionLabel>
            {parameters.length === 0 ? (
              <Text fontSize="sm" color={muted}>
                None
              </Text>
            ) : (
              parameters.map(([name, field]) => (
                <ParamRow key={`p-${name}`} name={name} field={field} />
              ))
            )}
          </Box>
        </>
      )}
    </VStack>
  );
};

const NodeCatalogDrawer: React.FC = () => {
  const { isOpen, close, initialNodeId } = useNodeCatalog();
  const { data, isLoading, error } = useUploadedNodes({ enabled: isOpen });
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const cardBg = useColorModeValue("white", "gray.800");
  const borderColor = useColorModeValue("#e5e5e5", "gray.700");
  const muted = useColorModeValue("gray.500", "gray.400");
  const hoverBg = useColorModeValue("#f5f5f5", "gray.700");
  const selectedBg = useColorModeValue("blue.50", "gray.700");
  const listBg = useColorModeValue("#f7f7f8", "gray.900");

  const nodes = (data?.nodes ?? []) as CatalogNode[];
  const categories = useMemo(() => uniqueCatalogCategories(nodes), [nodes]);
  const visible = useMemo(
    () => filterNodeCatalog(nodes, query, category),
    [nodes, query, category]
  );
  const grouped = useMemo(() => groupNodesByCategory(visible), [visible]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    if (initialNodeId) {
      setSelectedKey(initialNodeId);
      const match = nodes.find((n) => n.id === initialNodeId);
      if (match?.category) {
        setCategory("all");
        setQuery("");
      }
      return;
    }
    if (!selectedKey && visible.length > 0) {
      setSelectedKey(nodeCatalogKey(visible[0]));
    }
  }, [isOpen, initialNodeId, nodes, selectedKey, visible]);

  const selected = useMemo(() => {
    if (!selectedKey) {
      return visible[0] ?? null;
    }
    return (
      visible.find((n) => nodeCatalogKey(n) === selectedKey) ||
      nodes.find((n) => nodeCatalogKey(n) === selectedKey) ||
      null
    );
  }, [selectedKey, visible, nodes]);

  return (
    <Drawer
      isOpen={isOpen}
      placement="right"
      onClose={close}
      size="xl"
      autoFocus
      initialFocusRef={searchInputRef}
    >
      <DrawerOverlay />
      <DrawerContent bg={cardBg} maxW={{ base: "100vw", md: "900px" }}>
        <DrawerCloseButton />
        <DrawerHeader borderBottomWidth="1px" borderColor={borderColor} pr={12}>
          <Heading size="md">Node catalog</Heading>
          <Text fontSize="sm" color={muted} fontWeight="normal" mt={1}>
            Workflow node types: name, category, ports, and schema description.
            This is not a dataset browser.
          </Text>
          <Text fontSize="xs" color={muted} mt={1}>
            {visible.length} of {nodes.length} types
          </Text>
        </DrawerHeader>
        <DrawerBody p={0}>
          <Flex direction={{ base: "column", md: "row" }} h="100%">
            <Box
              w={{ base: "100%", md: "340px" }}
              borderRightWidth={{ base: 0, md: "1px" }}
              borderColor={borderColor}
              bg={listBg}
              display="flex"
              flexDirection="column"
              minH={{ base: "240px", md: "100%" }}
            >
              <Box p={3} borderBottomWidth="1px" borderColor={borderColor}>
                <InputGroup size="sm" mb={2}>
                  <InputLeftElement pointerEvents="none">
                    <SearchIcon color={muted} />
                  </InputLeftElement>
                  <Input
                    ref={searchInputRef}
                    placeholder="Search name, description, ports…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    bg={cardBg}
                  />
                </InputGroup>
                <Select
                  size="sm"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  bg={cardBg}
                >
                  <option value="all">All categories</option>
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </Select>
              </Box>
              <Box flex="1" overflowY="auto" p={2}>
                {isLoading && (
                  <HStack justify="center" py={8}>
                    <Spinner size="sm" />
                    <Text fontSize="sm" color={muted}>
                      Loading node types…
                    </Text>
                  </HStack>
                )}
                {error && (
                  <Alert status="error" borderRadius="md">
                    <AlertIcon />
                    {error}
                  </Alert>
                )}
                {!isLoading && !error && visible.length === 0 && (
                  <Text fontSize="sm" color={muted} p={3}>
                    {nodes.length === 0
                      ? "No node types are available for your account."
                      : "No node types match this search."}
                  </Text>
                )}
                {grouped.map(([cat, catNodes]) => (
                  <Box key={cat} mb={3}>
                    <Text
                      fontSize="xs"
                      fontWeight="bold"
                      color={muted}
                      px={2}
                      py={1}
                      textTransform="uppercase"
                    >
                      {cat}
                    </Text>
                    {catNodes.map((node) => {
                      const key = nodeCatalogKey(node);
                      const isSelected = selectedKey === key;
                      return (
                        <Box
                          key={key}
                          as="button"
                          type="button"
                          aria-current={isSelected ? "true" : undefined}
                          textAlign="left"
                          w="100%"
                          px={3}
                          py={2}
                          mb={1}
                          borderRadius="md"
                          bg={isSelected ? selectedBg : "transparent"}
                          _hover={{ bg: isSelected ? selectedBg : hoverBg }}
                          onClick={() => setSelectedKey(key)}
                        >
                          <Text fontSize="sm" fontWeight="semibold" noOfLines={1}>
                            {node.label}
                          </Text>
                          {node.description ? (
                            <Text fontSize="xs" color={muted} noOfLines={2}>
                              {node.description}
                            </Text>
                          ) : null}
                        </Box>
                      );
                    })}
                  </Box>
                ))}
              </Box>
            </Box>
            <Box flex="1" overflowY="auto" p={5}>
              {!isLoading && selected ? (
                <NodeDetail node={selected} />
              ) : !isLoading && !error ? (
                <Text color={muted}>Select a node type to see ports and parameters.</Text>
              ) : null}
            </Box>
          </Flex>
        </DrawerBody>
      </DrawerContent>
    </Drawer>
  );
};

export default NodeCatalogDrawer;
