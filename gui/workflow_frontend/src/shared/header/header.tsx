import {
  Flex,
  Heading,
  Button,
  Spacer,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Box,
  Avatar,
  HStack,
  Text,
  IconButton,
  Tooltip,
  useToast,
  useColorMode,
  useColorModeValue,
} from '@chakra-ui/react'
import { Link as RouterLink } from 'react-router-dom'
import { ChevronDownIcon, MoonIcon, SunIcon } from '@chakra-ui/icons'
import { useAuth } from '../../auth/authContext'
import { useNodeCatalog } from '../../views/home/nodeCatalog/NodeCatalogContext'

const Header: React.FC = () => {
  const { user, signOut } = useAuth();
  const { open: openNodeCatalog } = useNodeCatalog();
  const toast = useToast();
  const { colorMode, toggleColorMode } = useColorMode();

  // dark: original colors | light: OpenAI-inspired clean whites
  const headerBg      = useColorModeValue('#ffffff', 'gray.900');
  const headerBorder  = useColorModeValue('#e5e5e5', 'gray.700');
  const headerColor   = useColorModeValue('#1a1a1a', 'white');
  const menuBg        = useColorModeValue('white', 'gray.800');
  const menuBorder    = useColorModeValue('#e5e5e5', 'gray.700');
  const menuHoverBg   = useColorModeValue('#f5f5f5', 'gray.700');
  const subtextColor  = useColorModeValue('gray.500', 'gray.500');

  const handleSignOut = async () => {
    try {
      const result = await signOut();
      if (result.success) {
        toast({
          title: 'Logged out successfully',
          status: 'success',
          duration: 2000,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Logout failed',
          description: result.error?.message,
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (error) {
      toast({
        title: 'An error occurred',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  return (
    <Flex
      as="header"
      width="100%"
      py={4}
      px={6}
      alignItems="center"
      bg={headerBg}
      color={headerColor}
      borderBottom="1px solid"
      borderColor={headerBorder}
    >
      <Heading
        as={RouterLink}
        to="/"
        size="md"
        letterSpacing="tight"
        fontWeight="bold"
        cursor="pointer"
        color={headerColor}
      >
        Neuro-Workflow
      </Heading>

      <Text
        fontSize="xs"
        color={subtextColor}
        ml={3}
        fontFamily="mono"
        userSelect="all"
      >
        v{__APP_VERSION__}
        {__GIT_COMMIT_HASH__ !== "unknown" && ` (${__GIT_COMMIT_HASH__})`}
      </Text>

      <Box marginLeft={5} fontSize="sm">
        <Text>Brain/MINDS 2.0</Text>
        <Text>Okinawa Institute of Science and Technology / RIKEN CBS</Text>
      </Box>

      <Spacer />

      <Flex alignItems="center">
        {/* Color mode toggle */}
        <Tooltip label={colorMode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
          <IconButton
            aria-label="Toggle color mode"
            icon={colorMode === 'dark' ? <SunIcon /> : <MoonIcon />}
            onClick={toggleColorMode}
            variant="ghost"
            size="md"
            mx={2}
            color={headerColor}
          />
        </Tooltip>

        {/* Project Menu */}
        <Menu>
          <MenuButton
            as={Button}
            variant="ghost"
            size="md"
            mx={2}
            rightIcon={<ChevronDownIcon />}
            color={headerColor}
          >
            Project
          </MenuButton>
          <MenuList bg={menuBg} borderColor={menuBorder}>
            <MenuItem
              as={RouterLink}
              to="/file/new"
              bg={menuBg}
              color={headerColor}
              _hover={{ bg: menuHoverBg }}
            >
              New
            </MenuItem>
          </MenuList>
        </Menu>

        {/* Nodes Menu */}
        <Menu>
          <MenuButton
            as={Button}
            variant="ghost"
            size="md"
            mx={2}
            rightIcon={<ChevronDownIcon />}
            color={headerColor}
          >
            Nodes
          </MenuButton>
          <MenuList bg={menuBg} borderColor={menuBorder}>
            <MenuItem
              bg={menuBg}
              color={headerColor}
              _hover={{ bg: menuHoverBg }}
              onClick={() => openNodeCatalog()}
            >
              Node catalog
            </MenuItem>
            <MenuItem
              as={RouterLink}
              to="/box/upload"
              bg={menuBg}
              color={headerColor}
              _hover={{ bg: menuHoverBg }}
            >
              Upload
            </MenuItem>
          </MenuList>
        </Menu>

        {/* Settings Menu */}
        <Menu>
          <MenuButton
            as={Button}
            variant="ghost"
            size="md"
            mx={2}
            rightIcon={<ChevronDownIcon />}
            color={headerColor}
          >
            Settings
          </MenuButton>
          <MenuList bg={menuBg} borderColor={menuBorder}>
            <MenuItem
              as={RouterLink}
              to="/settings/databases"
              bg={menuBg}
              color={headerColor}
              _hover={{ bg: menuHoverBg }}
            >
              Custom Databases
            </MenuItem>
          </MenuList>
        </Menu>

        {/* User Menu */}
        {user && (
          <Menu>
            <MenuButton
              as={Button}
              variant="ghost"
              size="md"
              mx={2}
              rightIcon={<ChevronDownIcon />}
              color={headerColor}
            >
              <HStack spacing={2}>
                <Avatar
                  size="sm"
                  name={user.user_metadata?.name || user.email}
                  bg="brand.500"
                />
                <Text fontSize="sm" display={{ base: 'none', md: 'block' }}>
                  {user.user_metadata?.name || user.email?.split('@')[0]}
                </Text>
              </HStack>
            </MenuButton>
            <MenuList bg={menuBg} borderColor={menuBorder} minW="160px">
              <MenuItem
                as={RouterLink}
                to="/user"
                bg={menuBg}
                color={headerColor}
                _hover={{ bg: menuHoverBg }}
              >
                Profile
              </MenuItem>
              <MenuItem
                bg={menuBg}
                _hover={{ bg: 'red.500', color: 'white' }}
                color="red.400"
                onClick={handleSignOut}
              >
                Logout
              </MenuItem>
            </MenuList>
          </Menu>
        )}
      </Flex>
    </Flex>
  )
}

export default Header
