# ==============================================================================
# CloudMask AI - Multi-Stage Production Frontend Container
# ==============================================================================
# Stage 1: Build React 19 + Three.js production bundle
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve via Nginx with reverse proxy to FastAPI backend
FROM nginx:alpine

# Remove default configuration and copy production configuration
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
