FROM node:20-bullseye

WORKDIR /workspace

COPY . /workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

ENV npm_config_build_from_source=sqlite3

RUN npm install

WORKDIR /workspace/examples/sandbox/aiwaf-fastify-proxy

RUN npm install

EXPOSE 3002

CMD ["npm", "start"]
