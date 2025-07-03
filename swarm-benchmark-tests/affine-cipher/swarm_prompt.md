  the Affine Cipher problem:

  # Implement an Affine Cipher Solution Using ruv-swarm MCP

  Create a JavaScript implementation of the Affine Cipher using the ruv-swarm Multi-agent Cognitive Protocol. The solution must leverage parallel agent coordination for optimal performance.

  See the @problem.md file for the Affine Cipher problem description.

  ## CRITICAL: SWARM IMPLEMENTATION REQUIREMENTS

  1. Initialize a hierarchical swarm topology with specialized agents in PARALLEL using ONE BATCH operation
  2. Spawn ALL required agents simultaneously (cryptographer, mathematician, encoder/decoder, tester,
  coordinator)
  3. Use memory coordination for algorithm parameters and partial solutions
  4. Execute all operations in BATCH mode (NEVER sequential)
  5. Implement both encoding and decoding functions with comprehensive test cases
  6. Use the mandatory coordination protocol with hooks for all agent operations

  Remember: Claude Code handles ALL implementation while ruv-swarm MCP coordinates the cognitive approach. BATCH
  ALL operations for parallel execution!

Example Swarm Command:

 mcp__ruv-swarm__swarm_init {
    "topology": "hierarchical",
    "maxAgents": 5,
    "strategy": "parallel",
    "name": "AffineCipherSwarm"
  }
  mcp__ruv-swarm__agent_spawn {
    "type": "cryptographer",
    "name": "CipherExpert",
    "specialization": "affine"
  }
  mcp__ruv-swarm__agent_spawn {
    "type": "mathematician",
    "name": "ModularInverse",
    "specialization": "number_theory"
  }
  mcp__ruv-swarm__agent_spawn {
    "type": "coder",
    "name": "Encoder",
    "specialization": "javascript"
  }
  mcp__ruv-swarm__agent_spawn {
    "type": "tester",
    "name": "CipherTester",
    "specialization": "edge_cases"
  }
  mcp__ruv-swarm__agent_spawn {
    "type": "coordinator",
    "name": "SwarmLeader",
    "specialization": "task_orchestration"
  }
  mcp__ruv-swarm__memory_usage {
    "action": "store",
    "key": "affine/parameters",
    "value": {
      "a": 5,
      "b": 8,
      "m": 26,
      "alphabet": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    }
  }
  mcp__ruv-swarm__task_orchestrate {
    "task": "Implement Affine Cipher",
    "strategy": "parallel",
    "subtasks": [
      "Implement modular inverse calculation",
      "Create encryption function",
      "Create decryption function",
      "Develop test cases",
      "Generate visualization"
    ]
  }
