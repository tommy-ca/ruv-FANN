# Changelog

## [1.0.16] - 2025-07-08

### Added
- **Secure Claude invocation** with explicit permission control
- **Secure npx command** with mandatory version pinning
- User confirmation prompts for elevated permissions
- Security notices for all potentially dangerous operations
- New `ruv-swarm-secure.js` binary as default

### Changed
- Default binary now points to secure version
- Legacy binary available as `ruv-swarm-legacy`
- All hook commands now display security notices

### Security
- Version pinning enforced for all npx commands
- Explicit permission model for Claude integration
- User consent required for elevated operations

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