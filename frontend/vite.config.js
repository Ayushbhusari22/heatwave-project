import {defineConfig} from 'vite'
import react from '@vitejs/plugin-react'
import {visualizer} from 'rollup-plugin-visualizer';

export default defineConfig({
  // base: '/heatwave-project/',
  plugins: [react(),visualizer()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/,'')
      }
    }
  },
  build: {
    chunkSizeWarningLimit: 600
  }
})
