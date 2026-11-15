'use strict';

const vm = require('node:vm');

// fastify -> light-my-request depends on cookie@2, which is ESM-only ("type": "module").
// Jest 30 on Node >= 24.9 (vm.SourceTextModule#hasAsyncGraph) loads require()d ESM
// natively; babel-transforming it to CommonJS there makes it fail with
// "exports is not defined". Older Node (CI uses 20) cannot, so cookie must be transformed.
const nativeRequireEsm = typeof vm.SourceTextModule?.prototype.hasAsyncGraph === 'function';

module.exports = {
    globalTeardown: '<rootDir>/test/jest.teardown.js',
    forceExit: true,
    coverageProvider: 'v8',
    transform: {
        '^.+\\.m?js$': ['babel-jest', { plugins: ['@babel/plugin-transform-modules-commonjs'] }]
    },
    transformIgnorePatterns: nativeRequireEsm
        ? ['/node_modules/']
        : ['node_modules[/\\\\](?!(?:.*[/\\\\])?cookie[/\\\\])'],
    testPathIgnorePatterns: ['<rootDir>/test/wasm-adapter-real.test.js'],
    collectCoverageFrom: ['index.js', 'train.js', 'lib/**/*.js', 'utils/**/*.js'],
    coverageThreshold: {
        global: { branches: 66, functions: 100, lines: 80, statements: 80 }
    }
};
