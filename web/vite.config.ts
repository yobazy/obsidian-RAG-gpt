import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = process.env.RAG_API ?? "http://127.0.0.1:8790";
const proxied = ["/ask", "/search", "/notes", "/status", "/ingest"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT ?? 5190),
    strictPort: true,
    proxy: Object.fromEntries(proxied.map((path) => [path, { target: api, changeOrigin: true }])),
  },
});
