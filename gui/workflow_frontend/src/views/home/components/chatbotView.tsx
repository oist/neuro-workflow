import KeywordSearch from '@/shared/keyWordSearch/keyWordSearch';
import {
  VStack,
  Box,
  Text,
  SimpleGrid,
  Icon,
  Heading,
  Divider,
  Spinner,
  IconButton,
  HStack,
  useToast,
  Input,
  Button,
  useDisclosure,
  Tooltip,
  Collapse,
  Badge,
} from '@chakra-ui/react';
import { useEffect, useState, useRef } from 'react';
import { IconType } from 'react-icons';
import { FiBox, FiCopy, FiTrash2, FiInfo, FiCode, FiRefreshCw, FiChevronDown, FiChevronRight, FiMenu } from 'react-icons/fi'; // デフォルトアイコンとして使用
import { IoChatboxEllipses } from "react-icons/io5";
import { SchemaFields } from '../home/type';
import { createAuthHeaders } from '../../api/authHeaders';

interface ChatbotProps {
  isLoading?: boolean;
  error?: string;
}

const ChatbotArea: React.FC<ChatbotProps> = ({isLoading = false, error}) => {
  const toast = useToast();

  // チャットボット開閉管理
  const [isChatbotExpand, setIsChatbotExpand] = useState<boolean>(false);

  return (
    <Box
        position="absolute"
        top="268px"
        left="8px"
    >
      <IconButton
        position="absolute"
        top="calc(100vh - 330px)"
        left="36px"
        zIndex={900}
        aria-label="メニュー開閉"
        icon={<IoChatboxEllipses />}
        onClick={() => setIsChatbotExpand(!isChatbotExpand)}
        colorScheme="gray"
        bg="gray.200"
        _hover={{ bg: 'gray.600' }}
      />
      <Box
        position="absolute"
        left={0}
        top="64px"
        height="calc(100vh - 348px)"
        width="320px"
        // 幅は isOpen によって変化。transition で滑らかに
        //width={isChatbotExpand ? '320px' : '8px'}
        display={isChatbotExpand ? 'block' : 'none'}
        transition="width 200ms ease"
        bg="gray.900"
        color="white"
        borderRight="1px solid"
        borderColor="gray.700"
        zIndex={10}
        flex="1"
        flexDirection="column"
      >
        <Box 
          p={4}
          overflowY="auto"
          height="100%"
          css={{
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              width: '8px',
              background: '#2D3748',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#4A5568',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb:hover': {
              background: '#718096',
            },
          }}
        >
          <VStack spacing={6} align="stretch">
              <Box
                marginTop="60px"
              >
                <Text
                  fontSize="sm"
                  fontWeight="bold"
                  color="gray.400"
                  textTransform="uppercase"
                  letterSpacing="wider"
                >
                  Chatbot
                </Text>
              </Box>
              <Box
                height="800px"
              >
                <iframe
                  src="https://chakra-ui.com"
                  width="100%"
                  height="100%"
                  style={{ border: "none" }}
                  title="Chakra UI Site"
                />
              </Box>
          </VStack>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatbotArea;
