import {
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  MenuDivider,
  Button,
  Text,
  Flex,
  useColorModeValue,
} from "@chakra-ui/react";
import { FiChevronDown, FiSliders } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import { useChatProfileStore } from "@/stores/chatProfileStore";

// Header dropdown to switch the chat profile (MCP tool allowlist + prompt).
// "Default" means no profile: all tools and the default prompt.
const ChatProfileSelector: React.FC = () => {
  const navigate = useNavigate();
  const profiles = useChatProfileStore((s) => s.profiles);
  const selectedProfileId = useChatProfileStore((s) => s.selectedProfileId);
  const selectProfile = useChatProfileStore((s) => s.selectProfile);

  const bg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('#e5e5e5', 'gray.600');
  const subtextColor = useColorModeValue('gray.500', 'gray.300');
  const hoverBg = useColorModeValue('#f5f5f5', 'gray.700');
  const activeBg = useColorModeValue('#ebebeb', 'gray.700');

  const selected = profiles.find((p) => p.id === selectedProfileId) ?? null;

  return (
    <Menu>
      <MenuButton
        as={Button}
        leftIcon={<FiSliders />}
        rightIcon={<FiChevronDown />}
        size="xs"
        variant="ghost"
        color={subtextColor}
        maxW="150px"
        fontWeight="normal"
        _hover={{ bg: hoverBg }}
        title="Chat profile (allowed tools & system prompt)"
      >
        <Text isTruncated fontSize="xs">
          {selected ? selected.name : "Default"}
        </Text>
      </MenuButton>
      <MenuList bg={bg} borderColor={borderColor} minW="220px" zIndex={2000}>
        <MenuItem
          fontSize="xs"
          bg={selected ? bg : activeBg}
          _hover={{ bg: hoverBg }}
          onClick={() => selectProfile(null)}
        >
          Default (all tools)
        </MenuItem>
        {profiles.length > 0 && <MenuDivider borderColor={borderColor} />}
        {profiles.map((profile) => (
          <MenuItem
            key={profile.id}
            fontSize="xs"
            bg={profile.id === selectedProfileId ? activeBg : bg}
            _hover={{ bg: hoverBg }}
            onClick={() => selectProfile(profile.id)}
          >
            <Flex justify="space-between" align="center" w="100%">
              <Text isTruncated maxW="140px">
                {profile.name}
              </Text>
              <Text fontSize="10px" color={subtextColor} flexShrink={0}>
                {profile.allowed_tools.length === 0
                  ? "no tools"
                  : `${profile.allowed_tools.length} tools`}
              </Text>
            </Flex>
          </MenuItem>
        ))}
        <MenuDivider borderColor={borderColor} />
        <MenuItem
          fontSize="xs"
          bg={bg}
          _hover={{ bg: hoverBg }}
          onClick={() => navigate("/settings/chat-profiles")}
        >
          Manage profiles…
        </MenuItem>
      </MenuList>
    </Menu>
  );
};

export default ChatProfileSelector;
