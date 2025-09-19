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
  useToast,
} from '@chakra-ui/react'
import { Link as RouterLink } from 'react-router-dom'
import { ChevronDownIcon } from '@chakra-ui/icons'
import { useAuth } from '../../auth/authContext'

const Header: React.FC = () => {
  const { user, signOut } = useAuth();
  const toast = useToast();

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
    <>
     <Flex
      as="header"
      width="100%"
      py={4}
      px={6}
      alignItems="center"
      bg="gray.900"
      color="white"
      position="fixed"  
      top="0"          
      zIndex="1000"    
    >
      <Heading 
          as={RouterLink} 
          to="/" 
          size="md" 
          letterSpacing="tight" 
          fontWeight="bold"
          cursor="pointer" 
        >
          Neuro-workflow
      </Heading>
      
      <Spacer />
      
      <Flex alignItems="center">
        {/* File Menu */}
        <Menu>
          <MenuButton 
            as={Button}
            variant="ghost" 
            colorScheme="white" 
            size="md" 
            mx={2}
            rightIcon={<ChevronDownIcon />}
          >
            File
          </MenuButton>
          <MenuList bg="gray.800" borderColor="gray.700">
            <MenuItem as={RouterLink} to="/file/new" bg="gray.800" _hover={{ bg: "gray.700" }}>New</MenuItem>
            {/* <MenuItem as={RouterLink} to="/file/open" bg="gray.800" _hover={{ bg: "gray.700" }}>Open</MenuItem>
            <MenuItem as={RouterLink} to="/file/save" bg="gray.800" _hover={{ bg: "gray.700" }}>Save</MenuItem>
            <MenuItem as={RouterLink} to="/file/close" bg="gray.800" _hover={{ bg: "gray.700" }}>Close</MenuItem> */}
          </MenuList>
        </Menu>

        {/* Box Menu */}
        <Menu>
          <MenuButton 
            as={Button}
            variant="ghost" 
            colorScheme="white" 
            size="md" 
            mx={2}
            rightIcon={<ChevronDownIcon />}
          >
            Box
          </MenuButton>
          <MenuList bg="gray.800" borderColor="gray.700">
            <MenuItem as={RouterLink} to="/box/upload" bg="gray.800" _hover={{ bg: "gray.700" }}>Upload</MenuItem>        
          </MenuList>
        </Menu>

        {/* Settings Menu */}
        <Menu>
          <MenuButton 
            as={Button}
            variant="ghost" 
            colorScheme="white" 
            size="md" 
            mx={2}
            rightIcon={<ChevronDownIcon />}
          >
            Settings
          </MenuButton>
          <MenuList bg="gray.800" borderColor="gray.700">
            <MenuItem as={RouterLink} to="/settings/server" bg="gray.800" _hover={{ bg: "gray.700" }}>Server</MenuItem>
          </MenuList>
        </Menu>

        {/* User Menu - Show only for authenticated users */}
        {user && (
          <Menu>
            <MenuButton 
              as={Button}
              variant="ghost" 
              colorScheme="white" 
              size="md" 
              mx={2}
              rightIcon={<ChevronDownIcon />}
            >
              <HStack spacing={2}>
                <Avatar
                  size="sm"
                  name={user.user_metadata?.name || user.email}
                  src={user.user_metadata?.avatar_url}
                  bg="brand.500"
                />
                <Text fontSize="sm" display={{ base: 'none', md: 'block' }}>
                  {user.user_metadata?.name || user.email?.split('@')[0]}
                </Text>
              </HStack>
            </MenuButton>
            <MenuList bg="gray.800" borderColor="gray.700" minW="160px">
              <MenuItem 
                bg="gray.800" 
                _hover={{ bg: "red.600" }}
                color="red.300"
                onClick={handleSignOut}
              >
                Logout
              </MenuItem>
            </MenuList>
          </Menu>
        )}
      </Flex>
    </Flex>
    <Box height="72px" /> {/* Spacer to prevent content from being hidden under fixed header */}
    </>    
  )
}

export default Header
