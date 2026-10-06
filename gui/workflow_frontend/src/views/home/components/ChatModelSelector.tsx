import {
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Button,
  Text,
  Flex,
  useColorModeValue,
} from "@chakra-ui/react";
import { FiChevronDown, FiCpu } from "react-icons/fi";
import { useChatModelStore } from "@/stores/chatModelStore";

// Header dropdown to switch the LLM model. Hidden unless the backend offers
// more than one model; the first one is the default.
const ChatModelSelector: React.FC = () => {
  const models = useChatModelStore((s) => s.models);
  const selectedModelId = useChatModelStore((s) => s.selectedModelId);
  const selectModel = useChatModelStore((s) => s.selectModel);

  const bg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('#e5e5e5', 'gray.600');
  const subtextColor = useColorModeValue('gray.500', 'gray.300');
  const hoverBg = useColorModeValue('#f5f5f5', 'gray.700');
  const activeBg = useColorModeValue('#ebebeb', 'gray.700');

  if (models.length < 2) return null;

  const activeId = selectedModelId ?? models[0].id;

  return (
    <Menu>
      <MenuButton
        as={Button}
        leftIcon={<FiCpu />}
        rightIcon={<FiChevronDown />}
        size="xs"
        variant="ghost"
        color={subtextColor}
        maxW="150px"
        fontWeight="normal"
        _hover={{ bg: hoverBg }}
        title="LLM model"
      >
        <Text isTruncated fontSize="xs">
          {activeId}
        </Text>
      </MenuButton>
      <MenuList bg={bg} borderColor={borderColor} minW="220px" zIndex={2000}>
        {models.map((model, index) => (
          <MenuItem
            key={model.id}
            fontSize="xs"
            bg={model.id === activeId ? activeBg : bg}
            _hover={{ bg: hoverBg }}
            onClick={() => selectModel(index === 0 ? null : model.id)}
          >
            <Flex justify="space-between" align="center" w="100%">
              <Text isTruncated maxW="140px">
                {model.id}
              </Text>
              <Text fontSize="10px" color={subtextColor} flexShrink={0}>
                {index === 0 ? "default · " : ""}
                {model.provider}
              </Text>
            </Flex>
          </MenuItem>
        ))}
      </MenuList>
    </Menu>
  );
};

export default ChatModelSelector;
