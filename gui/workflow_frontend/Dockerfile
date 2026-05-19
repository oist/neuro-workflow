FROM node:24-slim AS build

WORKDIR /frontend

RUN corepack enable

COPY package.json pnpm-lock.yaml /frontend/

RUN pnpm install --frozen-lockfile --ignore-scripts

COPY . /frontend

RUN pnpm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /frontend/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
