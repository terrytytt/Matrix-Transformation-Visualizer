import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: './' makes the built assets resolve relative to the HTML file,
// so the same build works on GitHub Pages under any repository path.
export default defineConfig({
  base: './',
  plugins: [react()],
  // The lazily-loaded three.js chunk is ~500 kB minified; that's expected.
  build: {
    chunkSizeWarningLimit: 600,
  },
});
