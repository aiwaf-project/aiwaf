function normalizeCountryList(value) {
  if (!value) return new Set();
  const values = typeof value === 'string' ? [value] : Array.from(value);
  return new Set(values
    .map(item => String(item || '').trim().toUpperCase())
    .filter(Boolean));
}

function evaluateGeoPolicy({
  country,
  allowCountries = [],
  blockCountries = [],
  dynamicBlocked = []
} = {}) {
  const normalizedCountry = String(country || '').trim().toUpperCase();
  if (!normalizedCountry) {
    return { shouldBlock: false, country: '', reason: '' };
  }

  const allow = normalizeCountryList(allowCountries);
  const block = normalizeCountryList(blockCountries);
  const dynamic = normalizeCountryList(dynamicBlocked);
  let shouldBlock;
  if (allow.size > 0) {
    shouldBlock = !allow.has(normalizedCountry);
  } else if (block.size > 0 || dynamic.size > 0) {
    shouldBlock = block.has(normalizedCountry) || dynamic.has(normalizedCountry);
  } else {
    shouldBlock = false;
  }

  return {
    shouldBlock,
    country: normalizedCountry,
    reason: shouldBlock ? `Geo blocked: ${normalizedCountry}` : ''
  };
}

module.exports = {
  normalizeCountryList,
  evaluateGeoPolicy
};
