# /github/epic/create

Implementation guide for creating GitHub epics in the ruv-FANN style.

## Quick Create

```bash
# Basic epic
gh issue create \
  --title "🚀 [EPIC] Your Epic Title" \
  --body "$(cat << 'EOF'
# 🚀 [EPIC] Your Epic Title

**Status**: 🆕 **PLANNING** | **Functionality**: 0% | **Test Coverage**: 0% | **Target**: 100%/100%

## 📋 Epic Overview

Your epic description here.

### 🎯 Key Objectives
- ✅ **Objective 1** - Description
- 🔄 **Objective 2** - Description
- 🆕 **Objective 3** - Description

EOF
)" \
  --label "epic,enhancement,priority: high"
```

## Full Epic Template

Save this as your epic body:

```markdown
# 🚀 [EPIC] {{TITLE}}

**Status**: 🆕 **PLANNING** | **Functionality**: 0% | **Test Coverage**: 0% | **Target**: 100%/100%

## 📋 Epic Overview

{{DESCRIPTION}}

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

### **1. Component Name** 🆕 PLANNED (0%)
Development tasks:
- [ ] **Architecture Design** - Define interfaces and data flow
- [ ] **Core Implementation** - Build main functionality
- [ ] **Unit Tests** - Achieve 95% coverage
- [ ] **Integration Tests** - Verify component interactions
- [ ] **Documentation** - API docs and examples
- [ ] **Performance Optimization** - Meet benchmarks

**Files**: `/src/component/*`

### **2. Another Component** 🆕 PLANNED (0%)
- [ ] **Task 1** - Description
- [ ] **Task 2** - Description
- [ ] **Task 3** - Description

**Files**: `/src/another/*`

---

## 🏛️ System Architecture

```
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
    │ Agent   │    │ Neural  │    │ Storage │
    │ Engine  │    │ Models  │    │ Layer   │
    └─────────┘    └─────────┘    └─────────┘
```

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

### **Week 1-2: Foundation & Planning**
- [ ] Complete architecture design
- [ ] Set up development environment
- [ ] Define all interfaces and contracts
- [ ] Create project scaffolding

### **Week 3-4: Core Development**
- [ ] Implement core components
- [ ] Build agent framework
- [ ] Integrate neural models
- [ ] Initial API implementation

### **Week 5-6: Testing & Integration**
- [ ] Complete unit test suite
- [ ] Integration testing
- [ ] Performance testing
- [ ] Bug fixes and optimization

### **Week 7-8: Documentation & Release**
- [ ] Complete API documentation
- [ ] User guides and tutorials
- [ ] Deployment preparation
- [ ] Final testing and release

---

## 🔗 Related Issues & Dependencies

### Subtasks (To Be Created)
- [ ] #TBD - Component 1 Implementation
- [ ] #TBD - Component 2 Implementation
- [ ] #TBD - Testing Suite Creation
- [ ] #TBD - Documentation Sprint

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

```
[░░░░░░░░░░░░░░░░░░░░] 0% - Planning Phase
```

### **Component Status**
- 🆕 All components in planning phase
- 🔄 0 components in progress
- ✅ 0 components complete

---

## 👥 Team Assignments

- **Epic Lead**: @{{USER}}
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

**Created**: {{DATE}}  
**Last Updated**: {{DATE}}  
**Target Completion**: {{TARGET_DATE}}  
**Risk Level**: 🟡 Medium  
**Priority**: 🔴 HIGH  

---

*This epic follows the ruv-FANN epic structure for comprehensive project tracking.*
```

## Step-by-Step Creation

### 1. Prepare Epic Content

```bash
# Set variables
EPIC_TITLE="Your Epic Title"
EPIC_DESC="Your epic description"
COMPONENTS="Component1,Component2,Component3"

# Create epic body file
cat > epic-body.md << EOF
[Insert template above with your values]
EOF
```

### 2. Create the Epic

```bash
# Create epic issue
gh issue create \
  --title "🚀 [EPIC] $EPIC_TITLE" \
  --body-file epic-body.md \
  --label "epic,enhancement,priority: high" \
  --assignee "@me"
```

### 3. Create Subtasks

```bash
# Get epic number
EPIC_NUM=123  # Replace with actual issue number

# Create subtasks
gh issue create \
  --title "📦 [SUBTASK] Component 1 Implementation" \
  --body "Part of epic #$EPIC_NUM" \
  --label "enhancement,subtask"

gh issue create \
  --title "📦 [SUBTASK] Component 2 Implementation" \
  --body "Part of epic #$EPIC_NUM" \
  --label "enhancement,subtask"
```

## With Swarm Coordination

Include swarm metadata:

```markdown
## 🐝 Swarm Coordination

**Swarm ID**: swarm-{{TIMESTAMP}}  
**Topology**: Hierarchical  
**Active Agents**: 0  
**Coordination Mode**: Planning  

### Swarm Agents (To Be Spawned)
- 🏗️ **Architect** - System design
- 💻 **Coders** - Implementation
- 🧪 **Testers** - Quality assurance
- 📚 **Documenter** - Documentation
- 🎯 **Coordinator** - Progress tracking
```

## Quick Commands

```bash
# List all epics
gh issue list --label "epic"

# View epic details
gh issue view 123

# Update epic status
gh issue edit 123 --body-file updated-epic.md

# Close completed epic
gh issue close 123
```