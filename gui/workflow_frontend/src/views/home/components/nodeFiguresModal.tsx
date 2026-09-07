import {
  Image,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  VStack,
} from "@chakra-ui/react";
import { NodeFigure } from "../../../stores/runStore";

interface NodeFiguresModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  figures: NodeFigure[];
}

// Full-size view of the figures a node emitted during the last run.
const NodeFiguresModal = ({ isOpen, onClose, title, figures }: NodeFiguresModalProps) => (
  <Modal isOpen={isOpen} onClose={onClose} size="4xl" scrollBehavior="inside">
    <ModalOverlay />
    <ModalContent>
      <ModalHeader>{title}</ModalHeader>
      <ModalCloseButton />
      <ModalBody pb={6}>
        <VStack spacing={4} align="stretch">
          {figures.map((fig) => (
            <Image
              key={fig.index}
              src={fig.src}
              alt={`figure ${fig.index + 1}`}
              maxW="100%"
              maxH="400px"
              objectFit="contain"
              my={2}
              borderRadius="md"
            />
          ))}
        </VStack>
      </ModalBody>
    </ModalContent>
  </Modal>
);

export default NodeFiguresModal;
