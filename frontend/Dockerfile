# frontend/Dockerfile

# --- Stage 1: Build static assets ---
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
# uncomment the following line if you have a package-lock.json file. For production builds.
# RUN npm ci  

#uncomment the following line if you don't have a package-lock.json file. For development builds.
RUN npm install

COPY . .
RUN npm run build

# --- Stage 2: Serve with Nginx ---
FROM nginx:alpine

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose HTTP port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]