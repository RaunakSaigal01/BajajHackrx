# Build stage
FROM node:18 as builder
WORKDIR /app
COPY . .
RUN npm install && npm run build

# Production stage
FROM node:18-slim
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
RUN npm install --omit=dev
CMD ["node", "dist/index.js"]
