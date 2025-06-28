#!/bin/bash
set -euo pipefail

# Veritas Nexus Release Automation Script
# Usage: ./scripts/release.sh [VERSION] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CARGO_TOML="$PROJECT_ROOT/Cargo.toml"
CHANGELOG="$PROJECT_ROOT/CHANGELOG.md"
README="$PROJECT_ROOT/README.md"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
    fi
    
    # Check if working directory is clean
    if ! git diff --quiet && ! git diff --cached --quiet; then
        log_error "Working directory is not clean. Please commit or stash changes."
    fi
    
    # Check for required tools
    for tool in cargo git; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
        fi
    done
    
    # Check if cargo-audit is available
    if ! command -v cargo-audit &> /dev/null; then
        log_warning "cargo-audit not found. Security audit will be skipped."
        log_warning "Install with: cargo install cargo-audit"
    fi
    
    log_success "Prerequisites check passed"
}

get_current_version() {
    grep '^version = ' "$CARGO_TOML" | sed 's/version = "\(.*\)"/\1/' | head -1
}

update_version() {
    local new_version="$1"
    local current_version
    current_version=$(get_current_version)
    
    log_info "Updating version from $current_version to $new_version"
    
    # Update Cargo.toml
    sed -i.bak "s/^version = \".*\"/version = \"$new_version\"/" "$CARGO_TOML"
    rm "$CARGO_TOML.bak"
    
    # Update README.md version references
    sed -i.bak "s/veritas-nexus = \"[^\"]*\"/veritas-nexus = \"$new_version\"/g" "$README"
    rm "$README.bak"
    
    log_success "Version updated to $new_version"
}

run_quality_checks() {
    log_info "Running quality checks..."
    
    # Check compilation
    log_info "Checking compilation..."
    if ! cargo check --all-targets --all-features; then
        log_error "Compilation check failed"
    fi
    
    # Run tests
    log_info "Running tests..."
    if ! cargo test --all-features; then
        log_error "Tests failed"
    fi
    
    # Run clippy
    log_info "Running clippy..."
    if ! cargo clippy --all-targets --all-features -- -D warnings; then
        log_error "Clippy check failed"
    fi
    
    # Check formatting
    log_info "Checking code formatting..."
    if ! cargo fmt --all -- --check; then
        log_error "Code formatting check failed. Run 'cargo fmt' to fix."
    fi
    
    # Build documentation
    log_info "Building documentation..."
    if ! cargo doc --all-features --no-deps; then
        log_error "Documentation build failed"
    fi
    
    # Security audit (if available)
    if command -v cargo-audit &> /dev/null; then
        log_info "Running security audit..."
        if ! cargo audit; then
            log_error "Security audit failed"
        fi
    fi
    
    log_success "Quality checks passed"
}

run_benchmarks() {
    log_info "Running benchmarks..."
    
    # Only run if benchmarking feature is available
    if cargo check --features benchmarking &> /dev/null; then
        cargo bench --features benchmarking
        log_success "Benchmarks completed"
    else
        log_warning "Benchmarks skipped (benchmarking feature not available)"
    fi
}

test_examples() {
    log_info "Testing examples..."
    
    local examples_dir="$PROJECT_ROOT/examples"
    if [ -d "$examples_dir" ]; then
        for example in "$examples_dir"/*.rs; do
            if [ -f "$example" ]; then
                local example_name
                example_name=$(basename "$example" .rs)
                log_info "Testing example: $example_name"
                
                if ! cargo check --example "$example_name"; then
                    log_error "Example $example_name failed to compile"
                fi
            fi
        done
        log_success "All examples tested"
    else
        log_warning "No examples directory found"
    fi
}

create_package() {
    log_info "Creating package..."
    
    # Clean previous builds
    cargo clean
    
    # Create package
    if ! cargo package; then
        log_error "Package creation failed"
    fi
    
    # List package contents for verification
    log_info "Package contents:"
    cargo package --list
    
    # Check package size
    local package_file
    package_file=$(find target/package -name "veritas-nexus-*.crate" | head -1)
    if [ -f "$package_file" ]; then
        local package_size
        package_size=$(stat -c%s "$package_file" 2>/dev/null || stat -f%z "$package_file" 2>/dev/null || echo "unknown")
        if [ "$package_size" != "unknown" ]; then
            local size_mb=$((package_size / 1024 / 1024))
            log_info "Package size: ${size_mb}MB"
            if [ "$size_mb" -gt 10 ]; then
                log_warning "Package size is larger than 10MB. Consider optimization."
            fi
        fi
    fi
    
    log_success "Package created successfully"
}

publish_package() {
    local dry_run="$1"
    
    if [ "$dry_run" = "true" ]; then
        log_info "Running publish dry-run..."
        cargo publish --dry-run
    else
        log_info "Publishing package to crates.io..."
        read -p "Are you sure you want to publish? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cargo publish
            log_success "Package published successfully"
        else
            log_info "Publication cancelled"
            return 1
        fi
    fi
}

create_git_tag() {
    local version="$1"
    local dry_run="$2"
    
    local tag="v$version"
    
    if [ "$dry_run" = "true" ]; then
        log_info "Would create git tag: $tag"
        return 0
    fi
    
    log_info "Creating git tag: $tag"
    
    # Commit version changes
    git add "$CARGO_TOML" "$README"
    git commit -m "Release version $version

🚀 Generated with Veritas Nexus release automation

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Create annotated tag
    git tag -a "$tag" -m "Release version $version"
    
    # Push to origin
    read -p "Push tag to origin? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin main
        git push origin "$tag"
        log_success "Tag pushed to origin"
    else
        log_info "Tag created locally only"
    fi
}

show_usage() {
    echo "Usage: $0 [VERSION] [--dry-run]"
    echo ""
    echo "Arguments:"
    echo "  VERSION     Semantic version (e.g., 0.1.0, 1.2.3-alpha.1)"
    echo "  --dry-run   Perform a dry run without actual publishing"
    echo ""
    echo "Examples:"
    echo "  $0 0.1.0-alpha.1 --dry-run"
    echo "  $0 0.1.0"
    echo ""
    echo "Environment Variables:"
    echo "  CARGO_REGISTRY_TOKEN  Token for crates.io publishing"
}

main() {
    local version=""
    local dry_run="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                ;;
            *)
                if [ -z "$version" ]; then
                    version="$1"
                else
                    log_error "Too many arguments"
                fi
                shift
                ;;
        esac
    done
    
    # Validate version
    if [ -z "$version" ]; then
        log_error "Version is required"
    fi
    
    # Validate semantic version format
    if ! echo "$version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$'; then
        log_error "Invalid semantic version format: $version"
    fi
    
    log_info "Starting release process for version $version"
    if [ "$dry_run" = "true" ]; then
        log_info "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Run release steps
    check_prerequisites
    
    if [ "$dry_run" = "false" ]; then
        update_version "$version"
    else
        log_info "Would update version to $version"
    fi
    
    run_quality_checks
    test_examples
    run_benchmarks
    create_package
    
    if ! publish_package "$dry_run"; then
        log_error "Publication failed or cancelled"
    fi
    
    create_git_tag "$version" "$dry_run"
    
    log_success "Release process completed successfully!"
    
    if [ "$dry_run" = "false" ]; then
        log_info "Post-release checklist:"
        log_info "  - Update documentation website"
        log_info "  - Announce release on GitHub"
        log_info "  - Monitor for issues and feedback"
        log_info "  - Update any dependent projects"
    fi
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi