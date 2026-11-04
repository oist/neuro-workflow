import { Box, Button, useColorMode } from "@chakra-ui/react";
import { readableSurface } from "../readableSurface";

export const NodeMenu = ({ 
  position, 
  onDelete, 
  onView, 
  onEdit, 
  onClose 
}: {
  position: { x: number, y: number },
  onDelete: () => void,
  onView: () => void,
  onEdit: () => void,
  onClose: () => void,
}) => {
  const { colorMode } = useColorMode();
  const surface = readableSurface(colorMode === "dark" ? "dark" : "light");
  return (
    <Box
      position="fixed" 
      left={`${position.x}px`}
      top={`${position.y}px`}
      zIndex={1000}
      bg={surface.bg}
      color={surface.text}
      borderWidth="1px"
      borderStyle="solid"
      borderColor={surface.border}
      borderRadius="md"
      boxShadow="lg"
      width="140px"
      overflow="hidden"
      onClick={(e) => e.stopPropagation()}
    >
      <Button
        width="100%"
        justifyContent="flex-start"
        borderRadius="0"
        variant="ghost"
        size="sm"
        onClick={() => {
          onView();
          onClose();
        }}
      >
        Node Information
      </Button>
      <Button
        width="100%"
        justifyContent="flex-start"
        borderRadius="0"
        variant="ghost"
        size="sm"
        colorScheme="red"
        onClick={() => {
          onDelete();
          onClose();
        }}
      >
        Delete
      </Button>
    </Box>
  );
};
