import { MantineProvider } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import { Workspace } from './components/Workspace';
import { theme } from './styles/theme';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import './styles/global.css';

export function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Notifications position="top-right" />
      <Workspace />
    </MantineProvider>
  );
}
