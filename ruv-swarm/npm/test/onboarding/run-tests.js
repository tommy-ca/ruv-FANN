#!/usr/bin/env node

/**
 * Test runner for onboarding modules
 * Run with: node test/onboarding/run-tests.js
 */

import { runOnboarding } from '../../src/onboarding/index.js';

async function testOnboardingFlow() {
  console.log('Testing onboarding flow with auto-accept mode...\n');

  try {
    // Test with auto-accept mode
    const result = await runOnboarding({
      autoAccept: true,
      verbose: true
    });

    console.log('\nOnboarding result:', result);

    if (result.success) {
      console.log('\n✅ Onboarding completed successfully!');
    } else {
      console.log('\n⚠️  Onboarding did not complete:', result.reason || result.error);
    }

  } catch (error) {
    console.error('\n❌ Onboarding test failed:', error);
    process.exit(1);
  }
}

// Run the test
testOnboardingFlow();