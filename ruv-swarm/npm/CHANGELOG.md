# Changelog

## [1.0.15] - 2025-07-08

### Security
- **CRITICAL**: Fixed supply chain attack vulnerability (Issue #107)
- Removed automatic `--dangerously-skip-permissions` flag injection
- Disabled all `npx ruv-swarm` command recommendations
- Added security warnings to documentation
- Added SECURITY.md with vulnerability details

### Changed
- Updated all documentation to remove unsafe `npx` usage
- Added prominent security warnings to CLAUDE.md
- Enhanced postinstall message to alert users of security update

### Added
- SECURITY.md file with CVE details and mitigation steps
- README.md with security best practices
- Comprehensive security warnings throughout documentation

## [1.0.14] - Previous version
- Contained critical security vulnerability
- DO NOT USE - upgrade to 1.0.15 immediately