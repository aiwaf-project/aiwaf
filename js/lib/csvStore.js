const fs = require('fs');
const path = require('path');

function ensureDir(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function escapeCsv(value) {
  const str = String(value ?? '');
  if (str.includes('"') || str.includes(',') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function parseCsvLine(line) {
  return (line.match(/("([^"]|"")*"|[^,]+)/g) || []).map(part => {
    const trimmed = String(part || '').trim();
    if (trimmed.startsWith('"') && trimmed.endsWith('"')) {
      return trimmed.slice(1, -1).replace(/""/g, '"');
    }
    return trimmed;
  });
}

function readRows(filePath, headers) {
  if (!fs.existsSync(filePath)) return [];

  const raw = fs.readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean);
  if (raw.length === 0) return [];

  const fileHeader = parseCsvLine(raw[0]);
  const startIndex = fileHeader.join(',') === headers.join(',') ? 1 : 0;
  const rows = [];

  for (let i = startIndex; i < raw.length; i += 1) {
    const cells = parseCsvLine(raw[i]);
    if (cells.length === 0) continue;
    const row = {};
    headers.forEach((header, index) => {
      row[header] = cells[index] ?? '';
    });
    rows.push(row);
  }

  return rows;
}

function writeRows(filePath, headers, rows) {
  ensureDir(filePath);
  const lines = [headers.join(',')];
  for (const row of rows) {
    lines.push(headers.map(header => escapeCsv(row[header] ?? '')).join(','));
  }
  fs.writeFileSync(filePath, `${lines.join('\n')}\n`, 'utf8');
}

function appendRow(filePath, headers, row) {
  ensureDir(filePath);
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, `${headers.join(',')}\n`, 'utf8');
  }
  const line = headers.map(header => escapeCsv(row[header] ?? '')).join(',');
  fs.appendFileSync(filePath, `${line}\n`, 'utf8');
}

function nextId(rows) {
  let max = 0;
  for (const row of rows) {
    const value = Number(row.id || 0);
    if (Number.isFinite(value) && value > max) max = value;
  }
  return max + 1;
}

module.exports = {
  readRows,
  writeRows,
  appendRow,
  nextId
};
