import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Flex, Box } from '@chakra-ui/react';
import Header from './shared/header/header';
import LoginView from './views/login/loginView';
import ProtectedRoute from './protectedRoute';
import { AuthProvider } from './auth/authContext';
import ReAuthGate from './auth/ReAuthGate';
import TabManager from './components/tabs/TabManager';

function Router() {
  return (
    <AuthProvider>
      <ReAuthGate />
      <BrowserRouter>
        <Routes>
          {/* Login page (no authentication required) */}
          <Route path="/login" element={<LoginView />} />

          {/* Pages that require authentication */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <Flex direction="column" h="100vh" overflow="hidden">
                  <Box flexShrink={0}>
                    <Header />
                  </Box>
                  <Box flex="1" minH="0" overflow="hidden" display="flex" flexDirection="column">
                    <TabManager />
                  </Box>
                </Flex>
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default Router;
