import React from 'react';
import {
  Box,
  HStack,
  Button,
  Text,
  IconButton,
  Tooltip,
  useColorModeValue,
} from '@chakra-ui/react';
import { CloseIcon } from '@chakra-ui/icons';
import { useTabContext } from './TabManager';

const TabBar: React.FC = () => {
  const { tabs, activeTabId, switchTab, closeTab } = useTabContext();

  return (
    <Box
      bg="gray.800"
      borderBottom="1px"
      borderColor="gray.700"
      py={0}
      px={2}
      minHeight="42px"
      display="flex"
      alignItems="center"
    >
      <HStack spacing={1} align="center" h="100%">
        {tabs.map((tab, index) => (
          <Box
            key={tab.id}
            position="relative"
            display="flex"
            alignItems="center"
          >
            <Box
              cursor="pointer"
              onClick={() => switchTab(tab.id)}
              bg={tab.isActive ? 'gray.700' : 'transparent'}
              borderTopRadius="6px"
              borderBottom="none"
              px={4}
              py={2}
              height="38px"
              display="flex"
              alignItems="center"
              gap={2}
              transition="all 0.2s ease-in-out"
              _hover={{
                bg: tab.isActive ? 'gray.700' : 'gray.750',
              }}
              borderTop="2px solid"
              borderLeft="1px solid"
              borderRight="1px solid"
              borderColor={tab.isActive ? 'blue.400' : 'gray.600'}
              borderTopColor={tab.isActive ? 'blue.400' : 'transparent'}
              maxW="250px"
              position="relative"
              zIndex={tab.isActive ? 2 : 1}
            >
              {/* タブアイコン */}
              <Text fontSize="sm">
                {tab.type === 'workflow' ? '🔬' : '📊'}
              </Text>
              
              {/* タブタイトル */}
              <Text
                fontSize="sm"
                fontWeight={tab.isActive ? '600' : '400'}
                color={tab.isActive ? 'white' : 'gray.300'}
                isTruncated
                maxW="160px"
              >
                {tab.title}
              </Text>

              {/* クローズボタン（JupyterLabタブのみ） */}
              {tab.type === 'jupyter' && (
                <Tooltip label="Close tab" fontSize="xs">
                  <IconButton
                    aria-label="Close tab"
                    icon={<CloseIcon />}
                    size="xs"
                    variant="ghost"
                    borderRadius="full"
                    w="20px"
                    h="20px"
                    minW="20px"
                    color={tab.isActive ? 'gray.300' : 'gray.400'}
                    _hover={{ 
                      bg: 'red.500',
                      color: 'white'
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      closeTab(tab.id);
                    }}
                  />
                </Tooltip>
              )}
            </Box>
          </Box>
        ))}
        
        {/* 空のスペース */}
        <Box flex="1" height="100%" />
      </HStack>
    </Box>
  );
};

export default TabBar;