//! Client IP resolution behind trusted proxies, shared by the Python and Node.js adapters.
//!
//! Proxy headers count only when the connecting peer is trusted. `X-Forwarded-For` is walked
//! from the right, stopping at the first address that isn't trusted, because proxies append
//! and anything further left is client-supplied. This matches `resolve_client_ip` in
//! `py/aiwaf/core/utils.py`, which remains as the fallback when this crate isn't installed.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Longer `X-Forwarded-For` values are ignored, in characters.
pub const MAX_FORWARDED_FOR_LENGTH: usize = 4096;
/// Only this many hops from the right are examined.
pub const MAX_FORWARDED_FOR_ENTRIES: usize = 32;

/// Trusted networks, parsed once into masks so each check is two integer operations.
#[derive(Clone, Debug, Default)]
pub struct TrustedProxies {
    v4: Vec<(u32, u32)>,
    v6: Vec<(u128, u128)>,
    invalid: Vec<String>,
}

/// Which proxy header an untrusted peer sent, for the misconfiguration warning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IgnoredHeader {
    ForwardedFor,
    RealIp,
}

impl IgnoredHeader {
    pub fn name(self) -> &'static str {
        match self {
            Self::ForwardedFor => "X-Forwarded-For",
            Self::RealIp => "X-Real-IP",
        }
    }
}

impl TrustedProxies {
    /// Parse addresses and CIDR ranges. Entries that don't parse are kept in `invalid` and
    /// never widen trust. Shortcut names such as "cloudflare" are expanded by the caller.
    pub fn new<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut out = Self::default();
        for entry in entries {
            let entry = entry.as_ref().trim();
            if entry.is_empty() {
                continue;
            }
            match parse_network(entry) {
                Some(Network::V4(network, mask)) => out.v4.push((network, mask)),
                Some(Network::V6(network, mask)) => out.v6.push((network, mask)),
                None => out.invalid.push(entry.to_string()),
            }
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.v4.is_empty() && self.v6.is_empty()
    }

    pub fn invalid(&self) -> &[String] {
        &self.invalid
    }

    pub fn contains(&self, ip: &IpAddr) -> bool {
        match ip {
            IpAddr::V4(v4) => {
                let value = u32::from(*v4);
                self.v4
                    .iter()
                    .any(|(network, mask)| value & mask == *network)
            }
            IpAddr::V6(v6) => {
                let value = u128::from(*v6);
                self.v6
                    .iter()
                    .any(|(network, mask)| value & mask == *network)
            }
        }
    }

    /// Resolve the client address. Returns the address and, when an untrusted peer sent proxy
    /// headers, which header was ignored. A missing peer means a local Unix socket, which only
    /// a co-located proxy can reach.
    pub fn resolve(
        &self,
        peer: &str,
        forwarded_for: &str,
        real_ip: &str,
    ) -> (String, Option<IgnoredHeader>) {
        let peer_text = py_strip(peer);
        let peer_ip = normalize_ip(peer_text);
        let fallback = peer_ip
            .as_ref()
            .map(|(_, text)| text.clone())
            .unwrap_or_else(|| peer_text.to_string());
        if self.is_empty() {
            return (fallback, None);
        }
        if !peer_text.is_empty() && !peer_ip.as_ref().is_some_and(|(ip, _)| self.contains(ip)) {
            let ignored = match (
                peer_ip.is_some(),
                forwarded_for.is_empty(),
                real_ip.is_empty(),
            ) {
                (true, false, _) => Some(IgnoredHeader::ForwardedFor),
                (true, true, false) => Some(IgnoredHeader::RealIp),
                _ => None,
            };
            return (fallback, ignored);
        }
        if !forwarded_for.is_empty()
            && forwarded_for.len() <= MAX_FORWARDED_FOR_LENGTH * 4
            && forwarded_for.chars().count() <= MAX_FORWARDED_FOR_LENGTH
        {
            let hops: Vec<&str> = forwarded_for.split(',').collect();
            let start = hops.len().saturating_sub(MAX_FORWARDED_FOR_ENTRIES);
            let mut rightmost: Option<String> = None;
            for hop in hops[start..].iter().rev() {
                // Stop rather than skip, so a malformed hop can't expose values to its left.
                let Some((ip, text)) = normalize_ip(hop) else {
                    break;
                };
                if !self.contains(&ip) {
                    return (text, None);
                }
                rightmost.get_or_insert(text);
            }
            // Every hop is internal. Only the rightmost was written by our proxy.
            if let Some(text) = rightmost {
                return (text, None);
            }
        }
        let resolved = normalize_ip(real_ip)
            .map(|(_, text)| text)
            .unwrap_or(fallback);
        (resolved, None)
    }
}

enum Network {
    V4(u32, u32),
    V6(u128, u128),
}

fn v4_mask(bits: u32) -> u32 {
    if bits == 0 {
        0
    } else {
        u32::MAX << (32 - bits)
    }
}

fn v6_mask(bits: u32) -> u128 {
    if bits == 0 {
        0
    } else {
        u128::MAX << (128 - bits)
    }
}

/// Parse like Python's `ipaddress.ip_network(entry, strict=False)`: an address, an address with a
/// prefix length, or for IPv4 an address with a dotted netmask or host mask.
fn parse_network(entry: &str) -> Option<Network> {
    let (address, prefix) = match entry.split_once('/') {
        Some((address, prefix)) => (address, Some(prefix)),
        None => (entry, None),
    };
    if prefix.is_some_and(|p| p.contains('/')) {
        return None;
    }
    let prefix_bits = |max: u32| -> Option<u32> {
        let p = prefix?;
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        p.parse::<u32>().ok().filter(|bits| *bits <= max)
    };
    if address.contains(':') {
        let ip: Ipv6Addr = address.parse().ok()?;
        let bits = if prefix.is_some() {
            prefix_bits(128)?
        } else {
            128
        };
        let mask = v6_mask(bits);
        return Some(Network::V6(u128::from(ip) & mask, mask));
    }
    let ip = parse_ipv4(address)?;
    let bits = match prefix {
        None => 32,
        Some(p) if p.bytes().all(|b| b.is_ascii_digit()) => prefix_bits(32)?,
        Some(p) => {
            // A dotted netmask, or failing that a host mask, as Python accepts.
            let mask = u32::from(parse_ipv4(p)?);
            if mask.leading_ones() + mask.trailing_zeros() == 32 {
                mask.leading_ones()
            } else if (!mask).leading_ones() + (!mask).trailing_zeros() == 32 {
                (!mask).leading_ones()
            } else {
                return None;
            }
        }
    };
    let mask = v4_mask(bits);
    Some(Network::V4(u32::from(ip) & mask, mask))
}

/// IPv4 in strict dotted-decimal form. Python rejects leading zeros, so this does too.
fn parse_ipv4(text: &str) -> Option<Ipv4Addr> {
    let mut octets = [0u8; 4];
    let mut parts = text.split('.');
    for octet in &mut octets {
        let part = parts.next()?;
        if part.is_empty() || part.len() > 3 || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if part.len() > 1 && part.starts_with('0') {
            return None;
        }
        *octet = part.parse().ok()?;
    }
    if parts.next().is_some() {
        return None;
    }
    Some(Ipv4Addr::from(octets))
}

/// Python's `str.strip()` also strips the information separators U+001C to U+001F.
fn py_strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// Canonical address text, or None when `value` isn't an IP literal. Accepts `[v6]`, `[v6]:port`
/// and `v4:port`; an IPv4-mapped IPv6 address becomes IPv4; an IPv6 zone ID is kept, as Python does.
pub fn normalize_ip(value: &str) -> Option<(IpAddr, String)> {
    let mut text = py_strip(value);
    if let (true, Some(end)) = (text.starts_with('['), text.find(']')) {
        text = &text[1..end];
    } else if text.matches(':').count() == 1 {
        text = text.split(':').next().unwrap_or("");
    }
    if text.contains(':') {
        let (address, scope) = match text.split_once('%') {
            Some((address, scope)) => {
                if scope.is_empty() || scope.contains('%') {
                    return None;
                }
                (address, Some(scope))
            }
            None => (text, None),
        };
        if !address
            .bytes()
            .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.')
        {
            return None;
        }
        let ip: Ipv6Addr = address.parse().ok()?;
        if let Some(v4) = ip.to_ipv4_mapped() {
            return Some((IpAddr::V4(v4), v4.to_string()));
        }
        let mut canonical = ip.to_string();
        if let Some(scope) = scope {
            canonical.push('%');
            canonical.push_str(scope);
        }
        return Some((IpAddr::V6(ip), canonical));
    }
    let ip = parse_ipv4(text)?;
    Some((IpAddr::V4(ip), ip.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULTS: [&str; 8] = [
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    ];

    fn defaults() -> TrustedProxies {
        TrustedProxies::new(DEFAULTS)
    }

    #[test]
    fn untrusted_peer_cannot_choose_its_address() {
        let (ip, ignored) = defaults().resolve("198.51.100.7", "127.0.0.1", "");
        assert_eq!(ip, "198.51.100.7");
        assert_eq!(ignored, Some(IgnoredHeader::ForwardedFor));
        let (_, ignored) = defaults().resolve("198.51.100.7", "", "127.0.0.1");
        assert_eq!(ignored, Some(IgnoredHeader::RealIp));
        assert_eq!(defaults().resolve("198.51.100.7", "", "").1, None);
    }

    #[test]
    fn walks_from_the_right_and_stops_at_the_first_untrusted_hop() {
        let trusted = defaults();
        assert_eq!(
            trusted.resolve("10.0.0.1", "203.0.113.5", "").0,
            "203.0.113.5"
        );
        assert_eq!(
            trusted
                .resolve("10.0.0.1", "127.0.0.1, 203.0.113.5, 10.0.0.2", "")
                .0,
            "203.0.113.5"
        );
        // A malformed hop stops the walk; only the rightmost internal hop is ours.
        assert_eq!(
            trusted
                .resolve("10.0.0.1", "203.0.113.5, junk, 10.0.0.2", "")
                .0,
            "10.0.0.2"
        );
        assert_eq!(
            trusted.resolve("10.0.0.1", "10.0.0.3, 10.0.0.2", "").0,
            "10.0.0.2"
        );
        assert_eq!(
            trusted.resolve("10.0.0.1", "", "203.0.113.9").0,
            "203.0.113.9"
        );
        assert_eq!(trusted.resolve("10.0.0.1", "junk", "").0, "10.0.0.1");
        assert_eq!(trusted.resolve("", "203.0.113.5", "").0, "203.0.113.5");
    }

    #[test]
    fn empty_trust_returns_the_peer() {
        let none = TrustedProxies::new(Vec::<String>::new());
        assert_eq!(
            none.resolve("10.0.0.1", "203.0.113.5", ""),
            ("10.0.0.1".to_string(), None)
        );
    }

    #[test]
    fn long_headers_and_hop_limit() {
        let trusted = defaults();
        let long = format!("{},203.0.113.5", "1".repeat(MAX_FORWARDED_FOR_LENGTH));
        assert_eq!(trusted.resolve("10.0.0.1", &long, "").0, "10.0.0.1");
        let mut hops = vec!["203.0.113.5"];
        hops.extend(std::iter::repeat("10.0.0.2").take(MAX_FORWARDED_FOR_ENTRIES));
        assert_eq!(
            trusted.resolve("10.0.0.1", &hops.join(","), "").0,
            "10.0.0.2"
        );
    }

    #[test]
    fn normalizes_like_python() {
        let n = |v: &str| normalize_ip(v).map(|(_, t)| t);
        assert_eq!(n(" 203.0.113.5:8080 ").as_deref(), Some("203.0.113.5"));
        assert_eq!(n("[2001:DB8::1]:443").as_deref(), Some("2001:db8::1"));
        assert_eq!(n("::ffff:203.0.113.5").as_deref(), Some("203.0.113.5"));
        assert_eq!(n("fe80::1%eth0").as_deref(), Some("fe80::1%eth0"));
        assert_eq!(n("010.0.0.1"), None);
        assert_eq!(n("unknown"), None);
        assert_eq!(n(""), None);
    }

    #[test]
    fn parses_networks_like_python() {
        let t = TrustedProxies::new([
            "10.0.0.0/255.0.0.0",
            "192.168.1.0/0.0.0.255",
            "2001:db8::/32",
            "203.0.113.9",
        ]);
        assert!(t.invalid().is_empty());
        assert!(t.contains(&"10.9.9.9".parse().unwrap()));
        assert!(t.contains(&"192.168.1.200".parse().unwrap()));
        assert!(t.contains(&"2001:db8::5".parse().unwrap()));
        assert!(t.contains(&"203.0.113.9".parse().unwrap()));
        assert!(!t.contains(&"203.0.113.10".parse().unwrap()));
        let bad = TrustedProxies::new([
            "10.0.0.0/33",
            "not-a-network",
            "10.0.0.0/8/1",
            "10.0.0.0/+8",
            "::/129",
        ]);
        assert!(bad.is_empty());
        assert_eq!(bad.invalid().len(), 5);
    }
}
