# Development Guidelines

## Versioning

- **Never** directly bump major, minor, or patch versions in a feature PR.
- Feature PRs must use the format `<version>-alpha.<revision>`, bumping the
  revision number relative to what is latest on `main`.
- Only release PRs may set a final release version (e.g., `1.1.0`).
