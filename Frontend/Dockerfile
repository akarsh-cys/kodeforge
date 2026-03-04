# Build from repo ROOT: docker build -f Frontend/Dockerfile -t codeforge-frontend \
#   --build-arg VITE_API_URL=https://<your-backend>.ap-south-1.awsapprunner.com .

FROM node:24-alpine AS builder
WORKDIR /app

# Receive backend URL at build time - Vite bakes this into the JS bundle
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

COPY Frontend/package*.json ./

# Install ALL deps (including devDeps like vite) so postinstall chmod succeeds
RUN npm install

# Copy source and build - VITE_API_URL is embedded into dist/ here
COPY Frontend/ .
RUN npm run build

# ── Lean runtime image (Vite SPA — served via nginx) ─────────────────
FROM nginx:1.27-alpine

# Copy Vite build output (dist/) into nginx web root
COPY --from=builder /app/dist /usr/share/nginx/html

# Use the repo's own nginx.conf (already has SPA fallback + error pages)
COPY Frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]