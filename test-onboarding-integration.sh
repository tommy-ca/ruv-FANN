#!/bin/bash

# Integration Test Script for Onboarding Components
# Tests both Rust and Node.js implementations

set -e

echo "🚀 Onboarding Integration Test Suite"
echo "====================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to log test results
log_test() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name - $message"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name - $message"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Function to run command with error handling
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_exit_code="${3:-0}"
    
    echo -e "${BLUE}Running:${NC} $test_name"
    echo "Command: $command"
    
    if eval "$command" > /tmp/test_output 2>&1; then
        local actual_exit_code=0
    else
        local actual_exit_code=$?
    fi
    
    if [ "$actual_exit_code" -eq "$expected_exit_code" ]; then
        log_test "$test_name" "PASS" "Exit code: $actual_exit_code"
        return 0
    else
        log_test "$test_name" "FAIL" "Expected exit code $expected_exit_code, got $actual_exit_code"
        echo "Output:"
        cat /tmp/test_output
        return 1
    fi
}

echo ""
echo "📋 Test 1: File Structure Validation"
echo "===================================="

# Test required files exist
required_files=(
    "/workspaces/ruv-swarm-cli/src/onboarding/mod.rs"
    "/workspaces/ruv-swarm-cli/ruv-swarm/npm/src/onboarding/index.js"
    "/workspaces/ruv-swarm-cli/ruv-swarm/crates/ruv-swarm-cli/src/commands/init.rs"
    "/workspaces/ruv-swarm-cli/ruv-swarm/crates/ruv-swarm-cli/src/main.rs"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        log_test "File Exists: $(basename $file)" "PASS" "Found at $file"
    else
        log_test "File Exists: $(basename $file)" "FAIL" "Missing: $file"
    fi
done

echo ""
echo "📋 Test 2: Integration Points Validation"
echo "========================================"

# Check if onboarding is properly integrated
if grep -q "skip_onboarding" "/workspaces/ruv-swarm-cli/ruv-swarm/crates/ruv-swarm-cli/src/main.rs"; then
    log_test "Rust CLI Skip Onboarding Flag" "PASS" "Flag properly integrated"
else
    log_test "Rust CLI Skip Onboarding Flag" "FAIL" "Flag not found in main.rs"
fi

if grep -q "runOnboarding" "/workspaces/ruv-swarm-cli/ruv-swarm/npm/src/index.js"; then
    log_test "Node.js Onboarding Export" "PASS" "Function properly exported"
else
    log_test "Node.js Onboarding Export" "FAIL" "Function not exported"
fi

echo ""
echo "📋 Test 3: Rust Compilation Test"
echo "================================"

# Test Rust code compiles
run_test "Rust Compilation Check" "cd /workspaces/ruv-swarm-cli && cargo check --quiet"

echo ""
echo "📊 Test Results Summary"
echo "======================"
echo -e "Total Tests: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! Integration successful.${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  Some tests failed. See details above.${NC}"
    exit 1
fi