import { MantineThemeOverride } from '@mantine/core';

export const theme: MantineThemeOverride = {
  fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  primaryColor: 'violet',
  colors: {
    dark: [
      '#C9D1D9',
      '#B1BAC4',
      '#8B949E',
      '#6E7681',
      '#484F58',
      '#30363D',
      '#21262D',
      '#161B22',
      '#0D1117',
      '#010409',
    ],
  },
  defaultRadius: 'md',
  shadows: {
    xs: '0 1px 3px rgba(0, 0, 0, 0.3)',
    sm: '0 2px 8px rgba(0, 0, 0, 0.4)',
    md: '0 4px 16px rgba(0, 0, 0, 0.5)',
    lg: '0 8px 32px rgba(0, 0, 0, 0.6)',
    xl: '0 16px 48px rgba(0, 0, 0, 0.7)',
  },
};
