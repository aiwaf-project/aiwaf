# Contributors

Thank you to everyone who has helped improve AIWAF.

## Maintainer

- **Aayush Gauba** ([@aayushgauba](https://github.com/aayushgauba)): creator and maintainer

## Code

- **William Touzalin** ([@Will-777](https://github.com/Will-777)): the Django `aiwaf_list` command for blocked and exempted IPs and monitored keywords ([#2](https://github.com/aiwaf-project/aiwaf/pull/2), [#3](https://github.com/aiwaf-project/aiwaf/pull/3)), IPv6 log parsing, and saving RequestLog entries from `AIAnomalyMiddleware` ([#27](https://github.com/aiwaf-project/aiwaf/pull/27))
- **Brady Endres** ([@soccer99](https://github.com/soccer99)): a more complete `.gitignore` ([#4](https://github.com/aiwaf-project/aiwaf/pull/4)) and test organization ([#5](https://github.com/aiwaf-project/aiwaf/pull/5))
- **Abhijeet Malviya** ([@abhi16apr](https://github.com/abhi16apr)): faster Java path normalization, with regression tests ([#29](https://github.com/aiwaf-project/aiwaf/pull/29))
- [@rene9ate](https://github.com/rene9ate): replaced bare `except:` clauses with `except Exception:`, sent the Flask GeoIP summary through the logger, and added a test that guards against new bare excepts ([#43](https://github.com/aiwaf-project/aiwaf/pull/43))
- [@7thplanetyt](https://github.com/7thplanetyt): Java access-log parsing that keeps the first client field, so IPv6 clients are no longer dropped, with regression tests ([#45](https://github.com/aiwaf-project/aiwaf/pull/45))
- **Aryan Gupta** ([@aryangup451-del](https://github.com/aryangup451-del)): `PATCH` support in the Python and Rust access-log parsers, so those requests are no longer dropped from training ([#36](https://github.com/aiwaf-project/aiwaf/pull/36))
- **Hamidjon** ([@hamidbek06](https://github.com/hamidbek06)): Flask and FastAPI access logs that record the correct UTC time on non-UTC servers, with regression tests ([#39](https://github.com/aiwaf-project/aiwaf/pull/39))

## Reviews

- [@Knoor59](https://github.com/Knoor59): reviewed [#43](https://github.com/aiwaf-project/aiwaf/pull/43)
- [@roberthollinswork](https://github.com/roberthollinswork): reviewed [#45](https://github.com/aiwaf-project/aiwaf/pull/45)
- **Edward Lee** ([@edwardlee4948](https://github.com/edwardlee4948)): reviewed [#38](https://github.com/aiwaf-project/aiwaf/pull/38)
- **Emin Mirzayev** ([@eminmirzayev](https://github.com/eminmirzayev)): reviewed [#35](https://github.com/aiwaf-project/aiwaf/pull/35)

Licensing details for contributions originally made under the MIT license are in [NOTICE](NOTICE).
