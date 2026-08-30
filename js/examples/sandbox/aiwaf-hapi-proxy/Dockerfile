FROM node:20-bullseye

WORKDIR /workspace
COPY . /workspace

RUN apt-get update && apt-get install -y python3 make g++ \
  && rm -rf /var/lib/apt/lists/*

RUN npm_config_build_from_source=true npm install

WORKDIR /workspace/examples/sandbox/aiwaf-hapi-proxy

RUN npm_config_build_from_source=true npm install

EXPOSE 3003
CMD ["npm", "start"]
