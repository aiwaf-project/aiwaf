const path = require('path');

function isTrustedModelPath(modelPath, { defaultPath = null, allowCustom = false } = {}) {
  if (!modelPath) return false;
  if (allowCustom) return true;
  if (!defaultPath) return false;

  try {
    return path.resolve(modelPath) === path.resolve(defaultPath);
  } catch (err) {
    return false;
  }
}

module.exports = {
  isTrustedModelPath
};
