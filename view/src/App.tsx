import { MantineProvider } from '@mantine/core';
import { Workspace } from './components/Workspace';
import { theme } from './styles/theme';
import '@mantine/core/styles.css';

export function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Workspace />
    </MantineProvider>
  );
}
