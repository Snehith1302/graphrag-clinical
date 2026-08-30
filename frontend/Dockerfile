FROM node:18-alpine

WORKDIR /app

# Copy package config and install dependencies
COPY package.json .
RUN npm install

# Copy source files
COPY . .

# Expose Vite dev server port
EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
