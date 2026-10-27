import { defineConfig } from 'vite'

// The brain simulation runs in Python (brain/server.py) on port 8765.
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/models': 'http://127.0.0.1:8765',
      '/ws': { target: 'ws://127.0.0.1:8765', ws: true },
    },
  },
})
