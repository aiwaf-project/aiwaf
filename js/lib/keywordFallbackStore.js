const fs = require('fs');
const path = require('path');

class KeywordFallbackStore {
  constructor(storagePath) {
    this.storagePath = storagePath;
    this.keywords = {};
  }

  load() {
    if (!fs.existsSync(this.storagePath)) {
      this.keywords = {};
      return;
    }
    this.keywords = JSON.parse(fs.readFileSync(this.storagePath, 'utf8'));
  }

  save() {
    fs.mkdirSync(path.dirname(this.storagePath), { recursive: true });
    fs.writeFileSync(this.storagePath, JSON.stringify(this.keywords, null, 2), 'utf8');
  }

  add(keyword, count = 1) {
    this.load();
    const normalized = String(keyword || '');
    if (!normalized) return;
    this.keywords[normalized] = Number(this.keywords[normalized] || 0) + Number(count || 1);
    this.save();
  }

  top(limit = 10) {
    this.load();
    return Object.entries(this.keywords)
      .sort((a, b) => b[1] - a[1])
      .slice(0, Number(limit || 10));
  }
}

module.exports = KeywordFallbackStore;
