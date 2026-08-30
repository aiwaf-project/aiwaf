FROM node:20-bullseye

WORKDIR /workspace

COPY . /workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

ENV npm_config_build_from_source=sqlite3

RUN npm install

WORKDIR /workspace/examples/sandbox/aiwaf-proxy

RUN npm install

EXPOSE 3000

CMD ["npm", "start"]
