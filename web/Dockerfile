# syntax=docker/dockerfile:1
   
FROM node:18-alpine

WORKDIR /web

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build

CMD ["npm", "start"]