//! Stateless exemption matching and combined route execution planning.
use pyo3::prelude::*;
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    net::IpAddr,
    sync::{Mutex, OnceLock},
};

static PATTERNS: OnceLock<Mutex<HashMap<String, Regex>>> = OnceLock::new();
fn matches(value: &str, pattern: &str) -> Option<bool> {
    #[cfg(windows)]
    let normalized = value.replace('/', "\\").to_lowercase();
    #[cfg(windows)]
    let value = normalized.as_str();
    let mut cache = PATTERNS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()?;
    if !cache.contains_key(pattern) {
        let compiled = Regex::new(pattern).ok()?;
        if cache.len() >= 256 {
            cache.clear();
        }
        cache.insert(pattern.into(), compiled);
    }
    Some(cache.get(pattern)?.is_match(value))
}

#[pyfunction]
pub fn policy_path_exempt(
    py: Python<'_>,
    path: String,
    rules: Vec<(String, Option<String>)>,
    allow_prefix: bool,
) -> Option<bool> {
    py.allow_threads(|| {
        for (exact, wildcard) in rules {
            if let Some(pattern) = wildcard {
                if matches(&path, &pattern)? {
                    return Some(true);
                }
            } else {
                if path == exact {
                    return Some(true);
                }
                let prefix = exact.trim_end_matches('/');
                if allow_prefix
                    && !prefix.is_empty()
                    && (path == prefix || path.starts_with(&format!("{prefix}/")))
                {
                    return Some(true);
                }
            }
        }
        Some(false)
    })
}

pub fn ip_exempt(
    ip: &str,
    ips: &HashSet<String>,
    patterns: &[String],
    networks: &[(String, u8)],
) -> Option<bool> {
    if ips.contains(ip) {
        return Some(true);
    }
    for pattern in patterns {
        if matches(ip, pattern)? {
            return Some(true);
        }
    }
    let address = ip.split('%').next()?.parse::<IpAddr>().ok();
    for (network, prefix) in networks {
        let network = network.parse::<IpAddr>().ok()?;
        let found = match (address, network) {
            (Some(IpAddr::V4(a)), IpAddr::V4(b)) if *prefix <= 32 => {
                let mask = if *prefix == 0 {
                    0
                } else {
                    u32::MAX << (32 - prefix)
                };
                u32::from(a) & mask == u32::from(b) & mask
            }
            (Some(IpAddr::V6(a)), IpAddr::V6(b)) if *prefix <= 128 => {
                let mask = if *prefix == 0 {
                    0
                } else {
                    u128::MAX << (128 - prefix)
                };
                u128::from(a) & mask == u128::from(b) & mask
            }
            _ => false,
        };
        if found {
            return Some(true);
        }
    }
    Some(false)
}

#[pyfunction]
pub fn policy_ip_exempt(
    py: Python<'_>,
    ip: String,
    ips: HashSet<String>,
    patterns: Vec<String>,
    networks: Vec<(String, u8)>,
) -> Option<bool> {
    py.allow_threads(|| ip_exempt(&ip, &ips, &patterns, &networks))
}

#[pyfunction]
pub fn policy_path_rule_index(
    py: Python<'_>,
    path: String,
    prefixes: Vec<String>,
) -> Option<usize> {
    py.allow_threads(|| {
        let mut best = None;
        let mut best_length = 0;
        for (index, prefix) in prefixes.iter().enumerate() {
            let length = prefix.chars().count();
            if (path == prefix.trim_end_matches('/') || path.starts_with(prefix))
                && (best.is_none() || length > best_length)
            {
                best = Some(index);
                best_length = length;
            }
        }
        best
    })
}

#[pyfunction]
pub fn policy_route_plan(
    py: Python<'_>,
    path: String,
    rules: Vec<(String, HashSet<String>)>,
    names: Vec<String>,
    full: bool,
    exempt: HashSet<String>,
    required: HashSet<String>,
) -> (Option<usize>, HashSet<String>) {
    py.allow_threads(|| {
        // Caller supplies the compiled longest-prefix-first list; first tie wins.
        let index = rules.iter().position(|(prefix, _)| {
            path == prefix.trim_end_matches('/') || path.starts_with(prefix)
        });
        let disabled = index.map(|i| &rules[i].1);
        let enabled = names
            .into_iter()
            .filter(|name| {
                required.contains(name)
                    || !(full
                        || exempt.contains(name)
                        || disabled.is_some_and(|d| d.contains(name)))
            })
            .collect();
        (index, enabled)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(policy_path_exempt, m)?)?;
    m.add_function(wrap_pyfunction!(policy_ip_exempt, m)?)?;
    m.add_function(wrap_pyfunction!(policy_route_plan, m)?)?;
    m.add_function(wrap_pyfunction!(policy_path_rule_index, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> HashSet<String> {
        HashSet::new()
    }

    #[test]
    fn exact_ips_and_patterns_are_exempt() {
        let ips = HashSet::from(["203.0.113.5".to_string()]);
        assert_eq!(ip_exempt("203.0.113.5", &ips, &[], &[]), Some(true));
        let patterns = vec![r"^198\.51\.100\.\d+$".to_string()];
        assert_eq!(ip_exempt("198.51.100.7", &none(), &patterns, &[]), Some(true));
        assert_eq!(ip_exempt("198.51.101.7", &none(), &patterns, &[]), Some(false));
    }

    #[test]
    fn ipv4_networks_respect_the_prefix() {
        let networks = vec![("10.1.0.0".to_string(), 16)];
        assert_eq!(ip_exempt("10.1.200.3", &none(), &[], &networks), Some(true));
        assert_eq!(ip_exempt("10.2.0.1", &none(), &[], &networks), Some(false));
        let everything = vec![("0.0.0.0".to_string(), 0)];
        assert_eq!(ip_exempt("8.8.8.8", &none(), &[], &everything), Some(true));
    }

    #[test]
    fn ipv6_networks_and_zone_ids_are_supported() {
        let networks = vec![("2001:db8::".to_string(), 32), ("fe80::".to_string(), 10)];
        assert_eq!(ip_exempt("2001:db8:1::5", &none(), &[], &networks), Some(true));
        assert_eq!(ip_exempt("2001:db9::5", &none(), &[], &networks), Some(false));
        assert_eq!(ip_exempt("fe80::1%eth0", &none(), &[], &networks), Some(true));
    }

    #[test]
    fn address_families_and_oversized_prefixes_never_match() {
        let v6 = vec![("2001:db8::".to_string(), 32)];
        assert_eq!(ip_exempt("10.0.0.1", &none(), &[], &v6), Some(false));
        let oversized = vec![("10.0.0.1".to_string(), 33)];
        assert_eq!(ip_exempt("10.0.0.1", &none(), &[], &oversized), Some(false));
        let any = vec![("10.0.0.0".to_string(), 8)];
        assert_eq!(ip_exempt("not-an-ip", &none(), &[], &any), Some(false));
    }

    #[test]
    fn invalid_rules_defer_to_the_python_fallback() {
        assert_eq!(ip_exempt("10.0.0.1", &none(), &["(".to_string()], &[]), None);
        let bad_network = vec![("not-a-network".to_string(), 8)];
        assert_eq!(ip_exempt("10.0.0.1", &none(), &[], &bad_network), None);
    }

    #[test]
    fn matches_compiles_and_reuses_patterns() {
        assert_eq!(matches("abc", "^a"), Some(true));
        assert_eq!(matches("abc", "^a"), Some(true));
        assert_eq!(matches("abc", "^b"), Some(false));
        assert_eq!(matches("abc", "("), None);
    }
}
