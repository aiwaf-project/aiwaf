FROM node:20-bullseye

WORKDIR /workspace
COPY . /workspace

RUN apt-get update && apt-get install -y python3 make g++ \
  && rm -rf /var/lib/apt/lists/*

RUN npm_config_build_from_source=true npm install

WORKDIR /workspace/examples/sandbox/aiwaf-adonis-proxy

RUN npm_config_build_from_source=true npm install

EXPOSE 3007
CMD ["npm", "start"]
