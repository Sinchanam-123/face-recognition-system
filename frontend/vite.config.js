import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React dev server runs on :5173 and proxies API + video calls to the
// Flask backend on :5000, so the frontend code can just use relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:5000",
      "/video_feed": "http://localhost:5000",
    },
  },
});
