'use strict';

// AIWAF_PROFILE: one setting for sensible defaults.
// - balanced (the default) keeps every default as it is.
// - strict halves the rate limits and turns the method policy on.
// - monitor is a dry run: every check runs, and what it would block is logged instead, as
//   "AIWAF monitor: would block ...". Nothing is blocked and no IP is added to the blocklist.
// A setting given explicitly always wins over the profile.

const PROFILES = ['strict', 'balanced', 'monitor'];
let active = null;

function resolve(value) {
  const raw = value === undefined || value === null || value === '' ? process.env.AIWAF_PROFILE : value;
  const name = String(raw || 'balanced').trim().toLowerCase();
  if (!PROFILES.includes(name)) throw new Error(`AIWAF_PROFILE must be one of ${PROFILES.join(', ')}, not ${JSON.stringify(raw)}`);
  return name;
}

function activate(value) {
  active = resolve(value);
  return active;
}

function monitoring() {
  try {
    return (active || resolve()) === 'monitor';
  } catch (_error) {
    return false;
  }
}

// The defaults a profile sets, given Node's own defaults.
function defaultsFor(profile) {
  if (profile === 'strict') {
    return { MAX_REQ: 50, FLOOD_REQ: 100, AIWAF_METHOD_POLICY_ENABLED: true, AIWAF_SQL_INJECTION_MODE: 'block', AIWAF_INJECTION_RISK_MODE: 'block' };
  }
  if (profile === 'monitor') return { AIWAF_SQL_INJECTION_MODE: 'monitor', AIWAF_INJECTION_RISK_MODE: 'log' };
  return {};
}

function wouldBlock(method, path, ip, status, reason) {
  console.warn(`AIWAF monitor: would block ${method || '-'} ${path || '-'} from ${ip || '-'} with ${status} (${reason || 'blocked'})`);
}

// True in monitor mode, after logging the IP that would have been added to the blocklist.
function skipBlocklist(ip, reason) {
  if (!monitoring()) return false;
  console.warn(`AIWAF monitor: would add ${ip} to the blocklist (${reason})`);
  return true;
}

module.exports = { PROFILES, activate, defaultsFor, monitoring, resolve, skipBlocklist, wouldBlock };
