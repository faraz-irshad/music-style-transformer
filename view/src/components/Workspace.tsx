import { Grid, Box } from '@mantine/core';
import { Header } from './Header';
import { HeroWaveform } from './HeroWaveform';
import { InputPanel } from './InputPanel';
import { StylePanel } from './StylePanel';
import { OutputPanel } from './OutputPanel';

export function Workspace() {
  return (
    <Box
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(180deg, #0D1117 0%, #161B22 50%, #21262D 100%)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Header />
      
      <Box
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '48px 32px',
        }}
      >
        <Box style={{ maxWidth: '1600px', margin: '0 auto' }}>
          <Grid gutter="xl">
            <Grid.Col span={12}>
              <HeroWaveform />
            </Grid.Col>

            <Grid.Col span={{ base: 12, md: 4 }}>
              <Box
                p="xl"
                style={{
                  background: 'rgba(33, 38, 45, 0.6)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(139, 92, 246, 0.15)',
                  borderRadius: '16px',
                  boxShadow: '0 4px 24px rgba(0, 0, 0, 0.4)',
                  height: '100%',
                }}
              >
                <InputPanel />
              </Box>
            </Grid.Col>

            <Grid.Col span={{ base: 12, md: 4 }}>
              <Box
                p="xl"
                style={{
                  background: 'rgba(33, 38, 45, 0.6)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(139, 92, 246, 0.15)',
                  borderRadius: '16px',
                  boxShadow: '0 4px 24px rgba(0, 0, 0, 0.4)',
                  height: '100%',
                }}
              >
                <StylePanel />
              </Box>
            </Grid.Col>

            <Grid.Col span={{ base: 12, md: 4 }}>
              <Box
                p="xl"
                style={{
                  background: 'rgba(33, 38, 45, 0.6)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(139, 92, 246, 0.15)',
                  borderRadius: '16px',
                  boxShadow: '0 4px 24px rgba(0, 0, 0, 0.4)',
                  height: '100%',
                }}
              >
                <OutputPanel />
              </Box>
            </Grid.Col>
          </Grid>
        </Box>
      </Box>
    </Box>
  );
}
