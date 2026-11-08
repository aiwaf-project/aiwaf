//! Common Apache/ISO dates. Unsupported forms fall back to Python's full parser.
use pyo3::prelude::*;
type DateParts = (i32, u32, u32, u32, u32, u32, u32, Option<i32>);
fn number(value: &str) -> Option<u32> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}
fn clock(value: &str) -> Option<(u32, u32, u32, u32)> {
    let parts: Vec<_> = value.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let (seconds, fraction) = parts[2]
        .split_once('.')
        .or_else(|| parts[2].split_once(','))
        .unwrap_or((parts[2], ""));
    if parts[2].contains(['.', ',']) && fraction.is_empty() {
        return None;
    }
    if !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut micro = fraction.chars().take(6).collect::<String>();
    while micro.len() < 6 {
        micro.push('0');
    }
    Some((
        number(parts[0])?,
        number(parts[1])?,
        number(seconds)?,
        number(&micro)?,
    ))
}
pub fn parts(value: &str, apache: bool) -> Option<DateParts> {
    if !value.is_ascii() {
        return None;
    }
    if apache {
        let stamp = value.split_whitespace().next()?;
        let pieces: Vec<_> = stamp.split('/').collect();
        if pieces.len() == 3 {
            let (year, time) = pieces[2].split_once(':')?;
            if year.len() != 4 || time.contains(['.', ',']) {
                return None;
            }
            let month = [
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            ]
            .iter()
            .position(|m| *m == pieces[1].to_ascii_lowercase())? as u32
                + 1;
            let (h, m, s, us) = clock(time)?;
            if us != 0 {
                return None;
            }
            return Some((
                number(year)? as i32,
                month,
                number(pieces[0])?,
                h,
                m,
                s,
                0,
                None,
            ));
        }
    }
    if value.len() < 19 || value.as_bytes()[4] != b'-' || value.as_bytes()[7] != b'-' {
        return None;
    }
    let (year, month, day) = (
        number(&value[..4])?,
        number(&value[5..7])?,
        number(&value[8..10])?,
    );
    let time = &value[11..];
    let (time, offset) = if let Some(time) = time.strip_suffix('Z') {
        (time, Some(0))
    } else if let Some(i) = time.find(['+', '-']) {
        let zone = &time[i + 1..];
        let z: Vec<_> = zone.split(':').collect();
        if z.len() != 2 || z[0].len() != 2 || z[1].len() != 2 {
            return None;
        }
        let (h, m) = (number(z[0])?, number(z[1])?);
        if h >= 24 || m >= 60 {
            return None;
        }
        let seconds = (h * 3600 + m * 60) as i32 * if time.as_bytes()[i] == b'-' { -1 } else { 1 };
        (&time[..i], Some(seconds))
    } else {
        (time, None)
    };
    if time.len() < 8 || time.as_bytes()[2] != b':' || time.as_bytes()[5] != b':' {
        return None;
    }
    let (h, m, s, us) = clock(time)?;
    Some((year as i32, month, day, h, m, s, us, offset))
}
#[pyfunction]
fn training_date_parts(py: Python<'_>, dates: Vec<String>, apache: bool) -> Vec<Option<DateParts>> {
    py.allow_threads(|| dates.iter().map(|date| parts(date, apache)).collect())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(training_date_parts, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parts;

    #[test]
    fn apache_dates_parse_with_any_month_case() {
        for stamp in ["10/Oct/2026:12:00:05 +0000", "10/OCT/2026:12:00:05 +0000"] {
            let parsed = parts(stamp, true).unwrap();
            assert_eq!(
                (parsed.0, parsed.1, parsed.2, parsed.3, parsed.4, parsed.5, parsed.6),
                (2026, 10, 10, 12, 0, 5, 0)
            );
        }
    }

    #[test]
    fn apache_dates_reject_unknown_forms_so_python_can_fall_back() {
        for stamp in [
            "10/Foo/2026:12:00:05 +0000",
            "10/Oct/26:12:00:05 +0000",
            "10/Oct/2026:12:00:05.5 +0000",
            "10/Oct/2026:12:00 +0000",
            "10/Őct/2026:12:00:05 +0000",
            "",
        ] {
            assert_eq!(parts(stamp, true), None, "{stamp:?}");
        }
    }

    #[test]
    fn iso_dates_keep_their_utc_offset() {
        let cases = [
            ("2026-10-12T01:02:03Z", Some(0)),
            ("2026-10-12T01:02:03+05:30", Some(19800)),
            ("2026-10-12T01:02:03-05:00", Some(-18000)),
            ("2026-10-12T01:02:03", None),
        ];
        for (stamp, offset) in cases {
            assert_eq!(
                parts(stamp, false),
                Some((2026, 10, 12, 1, 2, 3, 0, offset)),
                "{stamp}"
            );
        }
    }

    #[test]
    fn iso_fractions_become_microseconds() {
        assert_eq!(parts("2026-10-12T01:02:03.5", false).unwrap().6, 500_000);
        assert_eq!(parts("2026-10-12T01:02:03,25", false).unwrap().6, 250_000);
        assert_eq!(parts("2026-10-12T01:02:03.1234567", false).unwrap().6, 123_456);
    }

    #[test]
    fn iso_dates_reject_malformed_values() {
        for stamp in [
            "2026-10-12",
            "2026/10/12T01:02:03",
            "2026-10-12T01:02:03+5:30",
            "2026-10-12T01:02:03+24:00",
            "2026-10-12T01:02:03+05:60",
            "2026-10-12T01:02:03.",
            "2026-10-12T01:02:xx",
        ] {
            assert_eq!(parts(stamp, false), None, "{stamp:?}");
        }
    }
}
