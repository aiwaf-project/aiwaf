use aiwaf_core::client_ip::TrustedProxies;
use pyo3::prelude::*;

/// Trusted proxy networks, parsed once. `resolve` runs per request.
#[pyclass(frozen)]
pub struct TrustedProxySet {
    inner: TrustedProxies,
}

#[pymethods]
impl TrustedProxySet {
    #[new]
    fn new(entries: Vec<String>) -> Self {
        Self {
            inner: TrustedProxies::new(entries),
        }
    }

    /// Entries that aren't an address or a CIDR range. They never widen trust.
    #[getter]
    fn invalid(&self) -> Vec<String> {
        self.inner.invalid().to_vec()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the client address and, when an untrusted peer sent proxy headers, the name of
    /// the header that was ignored.
    #[pyo3(signature = (peer, forwarded_for=None, real_ip=None))]
    fn resolve(
        &self,
        peer: Option<&str>,
        forwarded_for: Option<&str>,
        real_ip: Option<&str>,
    ) -> (String, Option<&'static str>) {
        let (peer, forwarded_for, real_ip) = (
            peer.unwrap_or(""),
            forwarded_for.unwrap_or(""),
            real_ip.unwrap_or(""),
        );
        let (ip, ignored) = self.inner.resolve(peer, forwarded_for, real_ip);
        (ip, ignored.map(|header| header.name()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TrustedProxySet>()
}
