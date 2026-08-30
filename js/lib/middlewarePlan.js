const AUTO_SENTINELS = new Set(['all', 'auto', 'aiwaf.all']);

function toList(value) {
  if (value === undefined || value === null) return null;
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    return value.split(',').map(item => item.trim()).filter(Boolean);
  }
  return null;
}

function isAutoSelection(requested) {
  const list = toList(requested);
  if (!list) return false;
  return list.some(item => AUTO_SENTINELS.has(String(item).trim().toLowerCase()));
}

function shouldEnableLogging(accessLog) {
  return !String(accessLog || '').trim();
}

function shouldEnableGeo({ geoEnabledFlag, staticBlockCountries, dynamicBlockCountries }) {
  if (geoEnabledFlag) return true;
  if ((staticBlockCountries || []).some(country => String(country || '').trim())) return true;
  if ((dynamicBlockCountries || []).some(country => String(country || '').trim())) return true;
  return false;
}

function shouldEnableUuidTamper({ hasUuidRoutes }) {
  if (hasUuidRoutes === undefined || hasUuidRoutes === null) return true;
  return !!hasUuidRoutes;
}

function planEnabledMiddlewares({
  orderedAvailable,
  requested,
  disabled,
  accessLog,
  geoEnabledFlag,
  staticBlockCountries,
  dynamicBlockCountries,
  hasUuidRoutes
}) {
  const available = new Set(orderedAvailable);
  const requestedList = toList(requested);
  const disabledSet = new Set((toList(disabled) || []).filter(name => available.has(name)));
  let enabled;

  if (!requestedList) {
    enabled = new Set(available);
  } else if (isAutoSelection(requestedList)) {
    enabled = new Set(available);
    if (!shouldEnableLogging(accessLog)) enabled.delete('logging');
    if (!shouldEnableGeo({ geoEnabledFlag, staticBlockCountries, dynamicBlockCountries })) {
      enabled.delete('geo_block');
    }
    if (!shouldEnableUuidTamper({ hasUuidRoutes })) enabled.delete('uuid_tamper');
  } else {
    enabled = new Set(requestedList.filter(name => available.has(name)));
  }

  disabledSet.forEach(name => enabled.delete(name));
  return enabled;
}

module.exports = {
  isAutoSelection,
  shouldEnableLogging,
  shouldEnableGeo,
  shouldEnableUuidTamper,
  planEnabledMiddlewares
};
