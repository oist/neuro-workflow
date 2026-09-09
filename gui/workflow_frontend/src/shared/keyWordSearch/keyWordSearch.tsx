import React, { useState, useCallback, useEffect } from 'react';
import {
  Input,
  InputGroup,
  InputRightElement,
  IconButton,
  Box,
  useColorModeValue,
} from '@chakra-ui/react';
import { SearchIcon } from '@chakra-ui/icons';

interface KeywordSearchProps {
  onSearch: (keyword: string) => void;
  placeholder?: string;
  size?: 'sm' | 'md' | 'lg';
  width?: string;
  variant?: string;
  bgColor?: string;
  borderRadius?: string;
  live?: boolean;
  liveDelayMs?: number;
}

const KeywordSearch: React.FC<KeywordSearchProps> = ({
  onSearch,
  placeholder = 'Search...',
  size = 'md',
  width = '100%',
  variant = 'filled',
  bgColor,
  borderRadius = 'md',
  live = false,
  liveDelayMs = 200,
}) => {
  const [keyword, setKeyword] = useState<string>('');
  const bg = useColorModeValue('gray.100', 'gray.700');
  const inputBg = bgColor || bg;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setKeyword(e.target.value);
  };

  const executeSearch = useCallback(() => {
    onSearch(keyword.trim());
  }, [keyword, onSearch]);

  useEffect(() => {
    if (!live) {
      return;
    }
    const id = window.setTimeout(() => {
      onSearch(keyword.trim());
    }, liveDelayMs);
    return () => window.clearTimeout(id);
  }, [keyword, live, liveDelayMs, onSearch]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      executeSearch();
    }
  };

  return (
    <Box width={width}>
      <InputGroup size={size}>
        <Input
          placeholder={placeholder}
          value={keyword}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          variant={variant}
          bg={inputBg}
          borderRadius={borderRadius}
          _focus={{
            borderColor: 'blue.400',
            boxShadow: '0 0 0 1px blue.400',
          }}
        />
        <InputRightElement>
          <IconButton
            aria-label="Search"
            icon={<SearchIcon />}
            size={size === 'lg' ? 'md' : 'sm'}
            onClick={executeSearch}
            variant="ghost"
            colorScheme="blue"
            borderRadius="full"
          />
        </InputRightElement>
      </InputGroup>
    </Box>
  );
};

export default KeywordSearch;
