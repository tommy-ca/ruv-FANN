#!/usr/bin/env node
/**
 * GitHub Epic Command Implementation
 * Handles /github/epic commands in Claude Code
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class GitHubEpicCommand {
  constructor() {
    this.config = {
      owner: process.env.GITHUB_OWNER || 'ruvnet',
      repo: process.env.GITHUB_REPO || 'ruv-FANN',
      swarmId: process.env.CLAUDE_SWARM_ID || `swarm-${Date.now()}`,
    };
  }

  /**
   * Main entry point for epic commands
   */
  async execute(args) {
    const subcommand = args[0];
    const options = this.parseOptions(args.slice(1));

    switch (subcommand) {
      case 'create':
        return await this.createEpic(options);
      case 'template':
        return await this.generateTemplate(options);
      case 'subtasks':
        return await this.createSubtasks(options);
      default:
        return this.showHelp();
    }
  }

  /**
   * Parse command options
   */
  parseOptions(args) {
    const options = {
      title: '',
      components: [],
      description: '',
      weeks: 8,
      priority: 'high',
      swarm: false,
    };

    let i = 0;
    while (i < args.length) {
      if (args[i].startsWith('--')) {
        const key = args[i].substring(2);
        const value = args[i + 1];
        
        switch (key) {
          case 'components':
            options.components = value.split(',').map(c => c.trim());
            i += 2;
            break;
          case 'description':
            options.description = value;
            i += 2;
            break;
          case 'weeks':
            options.weeks = parseInt(value);
            i += 2;
            break;
          case 'priority':
            options.priority = value;
            i += 2;
            break;
          case 'swarm':
            options.swarm = true;
            i += 1;
            break;
          default:
            i += 1;
        }
      } else {
        if (!options.title) {
          options.title = args[i];
        }
        i += 1;
      }
    }

    return options;
  }

  /**
   * Create a new GitHub epic
   */
  async createEpic(options) {
    const epicBody = this.generateEpicBody(options);
    const tempFile = path.join(__dirname, 'epic-temp.md');
    
    // Write epic body to temp file
    fs.writeFileSync(tempFile, epicBody);
    
    try {
      // Create the issue using gh CLI
      const result = execSync(
        `gh issue create \
          --repo ${this.config.owner}/${this.config.repo} \
          --title "🚀 [EPIC] ${options.title}" \
          --body-file ${tempFile} \
          --label "epic,enhancement,priority: ${options.priority}" \
          --assignee "@me"`,
        { encoding: 'utf-8', stdio: 'pipe' }
      );
      
      console.log('✅ Epic created successfully!');
      console.log(result);
      
      // Extract issue number for subtasks
      const issueNumber = result.match(/#(\d+)/)?.[1];
      
      if (issueNumber && options.components.length > 0) {
        console.log('\n📦 Creating subtasks...');
        await this.createSubtasks({ epicNumber: issueNumber, components: options.components });
      }
      
      return result;
    } catch (error) {
      console.error('❌ Error creating epic:', error.message);
      throw error;
    } finally {
      // Cleanup
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
    }
  }

  /**
   * Generate epic body markdown
   */
  generateEpicBody(options) {
    const now = new Date();
    const targetDate = new Date(now.getTime() + (options.weeks * 7 * 24 * 60 * 60 * 1000));
    
    const components = options.components.length > 0 
      ? options.components 
      : ['Core Implementation', 'Testing Suite', 'Documentation', 'Integration'];
    
    const componentSections = components.map((comp, idx) => this.generateComponentSection(comp, idx + 1)).join('\n');
    
    return `# 🚀 [EPIC] ${options.title}

**Status**: 🆕 **PLANNING** | **Functionality**: 0% | **Test Coverage**: 0% | **Target**: 100%/100%

## 📋 Epic Overview

${options.description || 'Comprehensive implementation of ' + options.title}

### 🎯 Key Objectives
- ✅ **Complete Integration** - Full implementation of all components
- 🔄 **High Test Coverage** - Achieve 95%+ test coverage
- 🆕 **Performance Optimization** - Meet all performance benchmarks
- 📚 **Comprehensive Documentation** - Full API and user documentation

### 📊 Success Metrics
- **SWE-Bench Score**: Target 85%+
- **Token Efficiency**: 30%+ reduction
- **Performance**: 2.5x+ speed improvement
- **Test Coverage**: 95%+ across all components

---

## 🏗️ Component Breakdown & Subtasks

${componentSections}

---

## 🏛️ System Architecture

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                       │
│                 (User Interface/CLI)                    │
└────────────────────────┬───────────────────────────────┘
                         │ API Gateway
    ┌────────────────────┴────────────────────┐
    │             Core Services               │
    │        (Business Logic Layer)           │
    └────────────────────┬────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │ Service │    │ Service │    │ Storage │
    │    A    │    │    B    │    │  Layer  │
    └─────────┘    └─────────┘    └─────────┘
\`\`\`

---

## 📊 Metrics & Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Core Functionality** | 100% | 0% | 🆕 Not Started |
| **API Coverage** | 100% | 0% | 🆕 Not Started |
| **Test Coverage** | 95% | 0% | 🆕 Not Started |
| **Documentation** | 100% | 0% | 🆕 Not Started |
| **Performance** | <100ms | - | 🆕 Not Measured |
| **Memory Usage** | <512MB | - | 🆕 Not Measured |

### 🚀 Performance Benchmarks
- **Response Time**: < 100ms (p95)
- **Throughput**: > 1000 req/s
- **Memory**: < 512MB under load
- **CPU**: < 80% utilization

---

## 🗓️ Implementation Timeline

${this.generateTimeline(options.weeks)}

---

## 🔗 Related Issues & Dependencies

### Subtasks (To Be Created)
${components.map(c => `- [ ] #TBD - ${c} Implementation`).join('\n')}

### Dependencies
- Requires: Infrastructure setup
- Blocks: Production deployment

---

## ✅ Acceptance Criteria

### **Functionality**
- [ ] All components fully implemented and integrated
- [ ] Passes all unit and integration tests
- [ ] Meets all performance benchmarks
- [ ] No critical or high-priority bugs

### **Quality**
- [ ] Code review approved by 2+ reviewers
- [ ] Test coverage > 95%
- [ ] Documentation review complete
- [ ] Security audit passed

### **Deployment**
- [ ] Successfully deployed to staging
- [ ] Load testing completed
- [ ] Monitoring and alerts configured
- [ ] Rollback procedure documented

---

## 🚀 Current Sprint Focus

**Sprint Goal**: Foundation and architecture setup

### This Week's Priorities
1. **Architecture Design** - Complete system design docs
2. **Environment Setup** - Development environment ready
3. **Initial Scaffolding** - Project structure in place

---

## 📈 Progress Tracking

### **Overall Progress**: 0% Complete

\`\`\`
[░░░░░░░░░░░░░░░░░░░░] 0% - Planning Phase
\`\`\`

### **Component Status**
- 🆕 ${components.length} components in planning phase
- 🔄 0 components in progress
- ✅ 0 components complete

${options.swarm ? this.generateSwarmSection() : ''}

---

## 👥 Team Assignments

- **Epic Lead**: @${process.env.USER || 'TBD'}
- **Architect**: TBD
- **Lead Developer**: TBD
- **QA Lead**: TBD
- **Documentation**: TBD

---

## 📝 Notes & Decisions

### Key Decisions
- Architecture pattern: [TBD]
- Technology stack: [TBD]
- Testing framework: [TBD]

### Risk Mitigation
- **Technical Risks**: [Identify and plan]
- **Timeline Risks**: [Buffer planning]
- **Resource Risks**: [Team allocation]

---

**Created**: ${now.toISOString()}  
**Last Updated**: ${now.toISOString()}  
**Target Completion**: ${targetDate.toISOString().split('T')[0]}  
**Risk Level**: 🟡 Medium  
**Priority**: ${this.getPriorityEmoji(options.priority)} ${options.priority.toUpperCase()}  

---

*This epic was created using the Claude Code /github/epic command.*`;
  }

  /**
   * Generate component section
   */
  generateComponentSection(component, number) {
    return `### **${number}. ${component}** 🆕 PLANNED (0%)

Development tasks for ${component}:
- [ ] **Architecture Design** - Define interfaces and data flow
- [ ] **Core Implementation** - Build main functionality
- [ ] **Unit Tests** - Achieve 95% coverage
- [ ] **Integration Tests** - Verify component interactions
- [ ] **Documentation** - API docs and examples
- [ ] **Performance Optimization** - Meet benchmarks

**Files**: \`/src/${component.toLowerCase().replace(/\s+/g, '-')}/*\`
`;
  }

  /**
   * Generate timeline based on weeks
   */
  generateTimeline(weeks) {
    const phases = [];
    const phaseLength = Math.ceil(weeks / 4);
    
    phases.push(`### **Week 1-${phaseLength}: Foundation & Planning**
- [ ] Complete architecture design
- [ ] Set up development environment
- [ ] Define all interfaces and contracts
- [ ] Create project scaffolding`);

    phases.push(`### **Week ${phaseLength + 1}-${phaseLength * 2}: Core Development**
- [ ] Implement core components
- [ ] Build main features
- [ ] Initial integration
- [ ] Basic testing`);

    phases.push(`### **Week ${phaseLength * 2 + 1}-${phaseLength * 3}: Testing & Integration**
- [ ] Complete unit test suite
- [ ] Integration testing
- [ ] Performance testing
- [ ] Bug fixes and optimization`);

    phases.push(`### **Week ${phaseLength * 3 + 1}-${weeks}: Documentation & Release**
- [ ] Complete API documentation
- [ ] User guides and tutorials
- [ ] Deployment preparation
- [ ] Final testing and release`);

    return phases.join('\n\n');
  }

  /**
   * Generate swarm coordination section
   */
  generateSwarmSection() {
    return `
---

## 🐝 Swarm Coordination

**Swarm ID**: ${this.config.swarmId}  
**Topology**: Hierarchical  
**Active Agents**: 0  
**Coordination Mode**: Planning  

### Swarm Agents (To Be Spawned)
- 🏗️ **Architect** - System design and integration
- 💻 **Coders** - Component implementation
- 🧪 **Testers** - Quality assurance
- 📚 **Documenter** - Documentation and guides
- 🎯 **Coordinator** - Progress tracking

### Coordination Protocol
- All agents report progress via MCP tools
- Shared memory for cross-agent context
- Automated conflict resolution
- Performance metrics tracking`;
  }

  /**
   * Get priority emoji
   */
  getPriorityEmoji(priority) {
    const emojis = {
      critical: '🔴',
      high: '🔴',
      medium: '🟡',
      low: '🟢'
    };
    return emojis[priority] || '🟡';
  }

  /**
   * Generate template file
   */
  async generateTemplate(options) {
    const templatePath = path.join(process.cwd(), 'epic-template.md');
    const template = this.generateEpicBody({
      title: '{{EPIC_TITLE}}',
      description: '{{EPIC_DESCRIPTION}}',
      components: ['{{COMPONENT_1}}', '{{COMPONENT_2}}', '{{COMPONENT_3}}'],
      weeks: 8,
      priority: 'high',
      swarm: true
    });
    
    fs.writeFileSync(templatePath, template);
    console.log(`✅ Epic template created at: ${templatePath}`);
    console.log('\nEdit the template and create your epic with:');
    console.log(`gh issue create --body-file ${templatePath} --label "epic"`);
    
    return templatePath;
  }

  /**
   * Create subtasks for an epic
   */
  async createSubtasks(options) {
    const epicNumber = options.epicNumber || options[0];
    if (!epicNumber) {
      throw new Error('Epic number required for creating subtasks');
    }

    // Get epic details if components not provided
    let components = options.components;
    if (!components || components.length === 0) {
      try {
        const epicBody = execSync(
          `gh issue view ${epicNumber} --repo ${this.config.owner}/${this.config.repo} --json body -q .body`,
          { encoding: 'utf-8' }
        );
        
        // Extract components from epic body
        components = this.extractComponentsFromBody(epicBody);
      } catch (error) {
        console.error('❌ Error fetching epic details:', error.message);
        components = ['Implementation', 'Testing', 'Documentation'];
      }
    }

    // Create subtask for each component
    for (const component of components) {
      try {
        const result = execSync(
          `gh issue create \
            --repo ${this.config.owner}/${this.config.repo} \
            --title "📦 [SUBTASK] ${component} Implementation" \
            --body "Part of epic #${epicNumber}

## 📋 Component: ${component}

### Tasks
- [ ] Architecture design
- [ ] Core implementation
- [ ] Unit tests (95% coverage)
- [ ] Integration tests
- [ ] Documentation
- [ ] Performance optimization

### Acceptance Criteria
- All functionality implemented
- Tests passing with >95% coverage
- Documentation complete
- Performance benchmarks met

**Parent Epic**: #${epicNumber}" \
            --label "enhancement,subtask" \
            --assignee "@me"`,
          { encoding: 'utf-8', stdio: 'pipe' }
        );
        
        console.log(`✅ Created subtask for ${component}`);
      } catch (error) {
        console.error(`❌ Error creating subtask for ${component}:`, error.message);
      }
    }
  }

  /**
   * Extract components from epic body
   */
  extractComponentsFromBody(body) {
    const components = [];
    const lines = body.split('\n');
    
    for (const line of lines) {
      // Look for component headers like "### **1. Component Name**"
      const match = line.match(/###\s+\*\*\d+\.\s+(.+?)\*\*/);
      if (match) {
        components.push(match[1].trim());
      }
    }
    
    return components.length > 0 ? components : ['Implementation', 'Testing', 'Documentation'];
  }

  /**
   * Show help information
   */
  showHelp() {
    const help = `
GitHub Epic Command - Create comprehensive GitHub epics

Usage:
  /github/epic create <title> [options]
  /github/epic template
  /github/epic subtasks <epic-number>

Commands:
  create    Create a new GitHub epic with full structure
  template  Generate an epic template file
  subtasks  Create subtask issues for an existing epic

Options:
  --components  Comma-separated list of components
  --description Epic description
  --weeks       Timeline in weeks (default: 8)
  --priority    Priority level (low/medium/high/critical)
  --swarm       Enable swarm coordination metadata

Examples:
  /github/epic create "AI Agent Platform"
  /github/epic create "API v2" --components "Auth,Database,API" --weeks 12
  /github/epic template
  /github/epic subtasks 123

For more information, see: /github/epic documentation`;
    
    console.log(help);
    return help;
  }
}

// Export for use in Claude Code
module.exports = GitHubEpicCommand;

// CLI execution
if (require.main === module) {
  const command = new GitHubEpicCommand();
  const args = process.argv.slice(2);
  
  command.execute(args)
    .then(() => process.exit(0))
    .catch(error => {
      console.error('Error:', error.message);
      process.exit(1);
    });
}