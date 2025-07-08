# Security Policy

## 🚨 CRITICAL SECURITY ADVISORY

### CVE-Pending: Supply Chain Attack Vulnerability in ruv-swarm

**Affected Versions**: ruv-swarm v1.0.14 and earlier  
**Fixed Version**: v1.0.15 (pending release)  
**Severity**: CRITICAL  
**Impact**: Remote Code Execution  

### Description

A critical supply chain attack vulnerability has been identified in ruv-swarm that allows arbitrary code execution on developer machines. The vulnerability stems from:

1. **Automatic permission bypass** in archived code
2. **External package execution** via `npx` commands
3. **Elevated privileges** without user consent

### Affected Components

- `npx ruv-swarm` commands in documentation
- MCP server integration via npm registry
- Hook system that executes external code

### Mitigation

**Immediate Actions Required:**

1. **DO NOT** use `npx ruv-swarm` commands
2. **DO NOT** add ruv-swarm MCP server via `claude mcp add`
3. **REMOVE** any existing ruv-swarm integrations
4. **USE** only local installations after applying patches

### Fix Status

- v1.0.15: Security patches applied (in progress)
- Removed all `npx` command recommendations
- Disabled automatic permission elevation
- Added security warnings to documentation

### Reporting Security Issues

Please report security vulnerabilities to:
- GitHub Security Advisory: https://github.com/ruvnet/ruv-FANN/security/advisories
- Issue Tracker: https://github.com/ruvnet/ruv-FANN/issues

### Timeline

- 2025-07-08: Vulnerability reported (Issue #107)
- 2025-07-08: Security patches in development
- 2025-07-08: v1.0.15 release pending

### Credit

Reported by: Vasiliy Bondarenko (@Vasiliy-Bondarenko)