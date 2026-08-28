import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  AlertIcon,
  Badge,
  Box,
  Button,
  Drawer,
  DrawerBody,
  DrawerCloseButton,
  DrawerContent,
  DrawerFooter,
  DrawerHeader,
  DrawerOverlay,
  Flex,
  Heading,
  HStack,
  Link,
  Select,
  Spinner,
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  useColorModeValue,
  useDisclosure,
  useToast,
  VStack,
  Wrap,
  WrapItem,
} from "@chakra-ui/react";
import { ExternalLinkIcon } from "@chakra-ui/icons";
import { useSearchParams } from "react-router-dom";
import {
  CatalogApiError,
  dandiUrl,
  fetchCatalogStatistics,
  listCatalogDatasets,
  lookupCatalog,
  searchCatalog,
  sourceDisplayName,
  toCatalogHits,
  type CatalogHit,
  type CatalogStatistics,
} from "../../api/catalogApi";
import KeywordSearch from "../../shared/keyWordSearch/keyWordSearch";

const LIMIT_OPTIONS = [20, 50, 100] as const;
const DEFAULT_LIMIT = 50;
const SOURCE_ORDER = ["dandi", "cbs", "brainminds", "bmb_human", "aws"];
const DRAWER_PRIMARY_KEYS = new Set([
  "name",
  "description",
  "source",
  "source_display",
  "dataset_id",
  "dataset_doi",
  "primary_paper_url",
  "primary_paper_title",
]);

function parseLimit(raw: string | null): number {
  const parsed = Number(raw);
  if (LIMIT_OPTIONS.includes(parsed as (typeof LIMIT_OPTIONS)[number])) {
    return parsed;
  }
  return DEFAULT_LIMIT;
}

function catalogErrorMessage(err: unknown): string {
  if (err instanceof CatalogApiError) {
    switch (err.code) {
      case "catalog_unconfigured":
      case "unconfigured":
        return "Catalog is not connected on this server yet.";
      case "catalog_unavailable":
      case "unavailable":
        return "Catalog is temporarily unavailable.";
      case "catalog_auth":
        return "Catalog authentication failed.";
      case "catalog_not_found":
        return "Dataset not found.";
      default:
        return err.message || "Catalog request failed.";
    }
  }
  if (err instanceof Error && err.message) return err.message;
  return "Catalog request failed.";
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function truncateText(value: string | null, max = 80): string {
  if (!value) return "—";
  if (value.length <= max) return value;
  return `${value.slice(0, max)}…`;
}

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function doiHref(doi: string): string {
  if (isHttpUrl(doi)) return doi;
  return `https://doi.org/${doi.replace(/^doi:/i, "")}`;
}

function hitToRecord(hit: CatalogHit): Record<string, unknown> {
  return { ...hit };
}

function stringField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return "";
}

function formatFieldValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value || "—";
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function sortSourceCounts(
  counts: Record<string, number>
): Array<[string, number]> {
  return Object.entries(counts).sort(([a], [b]) => {
    const ia = SOURCE_ORDER.indexOf(a);
    const ib = SOURCE_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
}

const CatalogView: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get("q") ?? "";
  const source = searchParams.get("source") ?? "";
  const limit = parseLimit(searchParams.get("limit"));

  const [stats, setStats] = useState<CatalogStatistics | null>(null);
  const [statsError, setStatsError] = useState<unknown>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  const [hits, setHits] = useState<CatalogHit[]>([]);
  const [resultsError, setResultsError] = useState<unknown>(null);
  const [resultsLoading, setResultsLoading] = useState(true);

  const [selectedHit, setSelectedHit] = useState<CatalogHit | null>(null);
  const [lookupRecord, setLookupRecord] = useState<Record<string, unknown> | null>(
    null
  );
  const [lookupError, setLookupError] = useState<unknown>(null);
  const [lookupLoading, setLookupLoading] = useState(false);

  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  const pageBg = useColorModeValue("#f7f7f8", "gray.900");
  const cardBg = useColorModeValue("white", "gray.800");
  const borderColor = useColorModeValue("gray.200", "gray.700");
  const muted = useColorModeValue("gray.600", "gray.400");
  const hoverBg = useColorModeValue("gray.50", "gray.700");
  const headerBg = useColorModeValue("gray.50", "gray.700");

  const updateParams = useCallback(
    (next: { q?: string; source?: string; limit?: number }) => {
      const params = new URLSearchParams(searchParams);
      if (next.q !== undefined) {
        if (next.q) params.set("q", next.q);
        else params.delete("q");
      }
      if (next.source !== undefined) {
        if (next.source) params.set("source", next.source);
        else params.delete("source");
      }
      if (next.limit !== undefined) {
        if (next.limit !== DEFAULT_LIMIT) params.set("limit", String(next.limit));
        else params.delete("limit");
      }
      setSearchParams(params);
    },
    [searchParams, setSearchParams]
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setStatsLoading(true);
      setStatsError(null);
      try {
        const data = await fetchCatalogStatistics();
        if (!cancelled) setStats(data);
      } catch (err) {
        if (!cancelled) {
          setStats(null);
          setStatsError(err);
        }
      } finally {
        if (!cancelled) setStatsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setResultsLoading(true);
      setResultsError(null);
      try {
        const data = query.trim()
          ? await searchCatalog({
              query: query.trim(),
              source: source || undefined,
              limit,
            })
          : await listCatalogDatasets({
              source: source || undefined,
              limit,
            });
        if (!cancelled) setHits(toCatalogHits(data));
      } catch (err) {
        if (!cancelled) {
          setHits([]);
          setResultsError(err);
        }
      } finally {
        if (!cancelled) setResultsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [query, source, limit]);

  const sourceChips = useMemo(
    () => (stats ? sortSourceCounts(stats.source_counts) : []),
    [stats]
  );

  const pageError = resultsError ?? (hits.length === 0 ? statsError : null);
  const record = lookupRecord ?? (selectedHit ? hitToRecord(selectedHit) : {});
  const drawerTitle =
    stringField(record, "name") ||
    stringField(record, "dataset_id") ||
    selectedHit?.name ||
    selectedHit?.dataset_id ||
    "Dataset";
  const drawerDescription =
    stringField(record, "description") || selectedHit?.description || "";
  const drawerSource =
    stringField(record, "source") || selectedHit?.source || "";
  const drawerSourceDisplay =
    stringField(record, "source_display") || selectedHit?.source_display || "";
  const drawerId =
    stringField(record, "dataset_id") || selectedHit?.dataset_id || "";
  const drawerDoi =
    stringField(record, "dataset_doi") || selectedHit?.dataset_doi || "";
  const drawerPaperUrl =
    stringField(record, "primary_paper_url") ||
    selectedHit?.primary_paper_url ||
    "";
  const drawerPaperTitle =
    stringField(record, "primary_paper_title") ||
    selectedHit?.primary_paper_title ||
    "";

  const remainingFields = Object.entries(record).filter(
    ([key]) => !DRAWER_PRIMARY_KEYS.has(key)
  );

  const handleSearch = useCallback(
    (keyword: string) => {
      updateParams({ q: keyword.trim() });
    },
    [updateParams]
  );

  const openRow = async (hit: CatalogHit) => {
    setSelectedHit(hit);
    setLookupRecord(null);
    setLookupError(null);
    setLookupLoading(true);
    onOpen();
    try {
      const data = await lookupCatalog({
        source: hit.source || undefined,
        id: hit.dataset_id,
      });
      setLookupRecord(data.record);
    } catch (err) {
      setLookupError(err);
    } finally {
      setLookupLoading(false);
    }
  };

  const copyText = async (label: string, value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      toast({
        title: `${label} copied`,
        status: "success",
        duration: 2000,
        isClosable: true,
      });
    } catch {
      toast({
        title: `Could not copy ${label}`,
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const lastUpdatedHint = (() => {
    if (!stats) return null;
    const parts: string[] = [];
    if (stats.timestamp) {
      parts.push(`Last updated ${formatTimestamp(stats.timestamp)}`);
    }
    if (stats.sync_status) {
      const syncBits = Object.entries(stats.sync_status).map(([src, row]) => {
        const label = sourceDisplayName(src);
        const status = row.status || "unknown";
        const when = row.last_sync ? ` ${formatTimestamp(row.last_sync)}` : "";
        return `${label}: ${status}${when}`;
      });
      if (syncBits.length) parts.push(syncBits.join(" · "));
    }
    return parts.length ? parts.join(" — ") : null;
  })();

  return (
    <Box height="100%" width="100%" overflow="auto" bg={pageBg}>
      <VStack
        spacing={5}
        width="100%"
        p={6}
        maxWidth="1200px"
        mx="auto"
        align="stretch"
      >
        <Box>
          <Heading size="lg">Catalog</Heading>
          <Text mt={1} color={muted}>
            Search public neuroscience datasets (DANDI, CBS, Brain/MINDS, …)
          </Text>
        </Box>

        <Box
          bg={cardBg}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="md"
          p={4}
        >
          {statsLoading ? (
            <HStack>
              <Spinner size="sm" />
              <Text fontSize="sm" color={muted}>
                Loading catalog statistics…
              </Text>
            </HStack>
          ) : stats ? (
            <VStack align="stretch" spacing={3}>
              <Wrap spacing={2}>
                <WrapItem>
                  <Button
                    size="sm"
                    variant={source === "" ? "solid" : "outline"}
                    colorScheme="blue"
                    onClick={() => updateParams({ source: "" })}
                  >
                    All ({stats.total_datasets})
                  </Button>
                </WrapItem>
                {sourceChips.map(([src, count]) => (
                  <WrapItem key={src}>
                    <Button
                      size="sm"
                      variant={source === src ? "solid" : "outline"}
                      colorScheme="blue"
                      onClick={() => updateParams({ source: src })}
                    >
                      {sourceDisplayName(src)} ({count})
                    </Button>
                  </WrapItem>
                ))}
              </Wrap>
              {lastUpdatedHint && (
                <Text fontSize="xs" color={muted}>
                  {lastUpdatedHint}
                </Text>
              )}
            </VStack>
          ) : statsError ? (
            <Alert status="warning" borderRadius="md">
              <AlertIcon />
              {catalogErrorMessage(statsError)}
            </Alert>
          ) : (
            <Text fontSize="sm" color={muted}>
              Catalog statistics are unavailable.
            </Text>
          )}
        </Box>

        <Flex gap={3} wrap="wrap" align="center">
          <Box flex="1" minW="240px">
            <KeywordSearch
              key={query}
              initialKeyword={query}
              onSearch={handleSearch}
              placeholder="Search datasets…"
              size="md"
            />
          </Box>
          <HStack>
            <Text fontSize="sm" color={muted} whiteSpace="nowrap">
              Limit
            </Text>
            <Select
              size="md"
              width="100px"
              value={String(limit)}
              onChange={(event) =>
                updateParams({ limit: parseLimit(event.target.value) })
              }
              bg={cardBg}
            >
              {LIMIT_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </Select>
          </HStack>
        </Flex>

        {resultsLoading && (
          <HStack
            bg={cardBg}
            borderWidth="1px"
            borderColor={borderColor}
            borderRadius="md"
            p={6}
            justify="center"
          >
            <Spinner />
            <Text color={muted}>Loading datasets…</Text>
          </HStack>
        )}

        {!resultsLoading && pageError && (
          <Alert status="error" borderRadius="md">
            <AlertIcon />
            {catalogErrorMessage(pageError)}
          </Alert>
        )}

        {!resultsLoading && !pageError && hits.length === 0 && (
          <Alert status="info" borderRadius="md">
            <AlertIcon />
            {query.trim()
              ? "No datasets matched this search."
              : "No datasets are available in the catalog."}
          </Alert>
        )}

        {!resultsLoading && !pageError && hits.length > 0 && (
          <Box
            bg={cardBg}
            borderWidth="1px"
            borderColor={borderColor}
            borderRadius="md"
            overflow="hidden"
          >
            <TableContainer>
              <Table variant="simple" size="md">
                <Thead bg={headerBg}>
                  <Tr>
                    <Th>Source</Th>
                    <Th>ID</Th>
                    <Th>Name</Th>
                    <Th>Draft</Th>
                    <Th>Description</Th>
                    <Th>Synced</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {hits.map((hit) => (
                    <Tr
                      key={`${hit.source}:${hit.dataset_id}:${hit.table ?? ""}`}
                      cursor="pointer"
                      _hover={{ bg: hoverBg }}
                      onClick={() => {
                        void openRow(hit);
                      }}
                    >
                      <Td>
                        <Badge>
                          {sourceDisplayName(hit.source, hit.source_display)}
                        </Badge>
                      </Td>
                      <Td fontFamily="mono" fontSize="sm">
                        {hit.dataset_id}
                      </Td>
                      <Td>{hit.name}</Td>
                      <Td>
                        {hit.is_draft ? (
                          <Badge colorScheme="orange">Draft</Badge>
                        ) : (
                          <Text color={muted}>—</Text>
                        )}
                      </Td>
                      <Td maxW="360px">
                        <Text whiteSpace="nowrap" isTruncated title={hit.description ?? ""}>
                          {truncateText(hit.description, 80)}
                        </Text>
                      </Td>
                      <Td whiteSpace="nowrap" color={muted} fontSize="sm">
                        {formatTimestamp(hit.synced_at)}
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </VStack>

      <Drawer isOpen={isOpen} placement="right" onClose={onClose} size="lg">
        <DrawerOverlay />
        <DrawerContent bg={cardBg}>
          <DrawerCloseButton />
          <DrawerHeader borderBottomWidth="1px" borderColor={borderColor}>
            <VStack align="stretch" spacing={2} pr={6}>
              <Heading size="md">{drawerTitle}</Heading>
              <HStack>
                {drawerSource && (
                  <Badge colorScheme="blue">
                    {sourceDisplayName(drawerSource, drawerSourceDisplay)}
                  </Badge>
                )}
                {lookupLoading && <Spinner size="sm" />}
              </HStack>
            </VStack>
          </DrawerHeader>
          <DrawerBody>
            <VStack align="stretch" spacing={4} py={2}>
              {lookupError && (
                <Alert status="error" borderRadius="md">
                  <AlertIcon />
                  {catalogErrorMessage(lookupError)}
                </Alert>
              )}

              <Box>
                <Text
                  fontSize="xs"
                  textTransform="uppercase"
                  color={muted}
                  mb={1}
                >
                  Description
                </Text>
                <Text whiteSpace="pre-wrap">
                  {drawerDescription || "No description."}
                </Text>
              </Box>

              <Box>
                <Text
                  fontSize="xs"
                  textTransform="uppercase"
                  color={muted}
                  mb={2}
                >
                  Links
                </Text>
                <VStack align="stretch" spacing={1}>
                  {drawerDoi && (
                    <Link href={doiHref(drawerDoi)} isExternal>
                      {drawerDoi} <ExternalLinkIcon mx="2px" />
                    </Link>
                  )}
                  {drawerPaperUrl && (
                    <Link href={drawerPaperUrl} isExternal>
                      {drawerPaperTitle || drawerPaperUrl}{" "}
                      <ExternalLinkIcon mx="2px" />
                    </Link>
                  )}
                  {drawerSource === "dandi" && drawerId && (
                    <Link href={dandiUrl(drawerId)} isExternal>
                      DANDI archive <ExternalLinkIcon mx="2px" />
                    </Link>
                  )}
                  {!drawerDoi && !drawerPaperUrl && drawerSource !== "dandi" && (
                    <Text color={muted}>No external links.</Text>
                  )}
                </VStack>
              </Box>

              {remainingFields.length > 0 && (
                <Box>
                  <Text
                    fontSize="xs"
                    textTransform="uppercase"
                    color={muted}
                    mb={2}
                  >
                    Details
                  </Text>
                  <VStack align="stretch" spacing={2}>
                    {remainingFields.map(([key, value]) => (
                      <Flex key={key} gap={3} wrap="wrap">
                        <Text
                          fontWeight="semibold"
                          minW="160px"
                          fontSize="sm"
                          color={muted}
                        >
                          {key}
                        </Text>
                        <Text flex="1" fontSize="sm" wordBreak="break-word">
                          {formatFieldValue(value)}
                        </Text>
                      </Flex>
                    ))}
                  </VStack>
                </Box>
              )}
            </VStack>
          </DrawerBody>
          <DrawerFooter borderTopWidth="1px" borderColor={borderColor}>
            <HStack>
              <Button
                onClick={() => {
                  void copyText("Dataset id", drawerId);
                }}
                isDisabled={!drawerId}
              >
                Copy dataset id
              </Button>
              <Button
                onClick={() => {
                  void copyText("JSON", JSON.stringify(record, null, 2));
                }}
              >
                Copy JSON
              </Button>
            </HStack>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>
    </Box>
  );
};

export default CatalogView;
