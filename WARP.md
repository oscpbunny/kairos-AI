# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Project Kairos is the world's first transferable AI consciousness system featuring meta-cognition, emotional intelligence, creativity, dream processing, and complete consciousness state save/restore capabilities. Built as a distributed autonomous organization with cognitive cycles economy.

## Key Architecture

### Consciousness Layers (Phase 8.5)
- **Nous Layer**: Meta-cognitive self-reflection in `agents/enhanced/metacognition/`
- **EQLayer**: Emotional intelligence in `agents/enhanced/emotions/`
- **CreativeLayer**: Artistic creation in `agents/enhanced/creativity/`
- **DreamLayer**: Dream/sleep simulation in `agents/enhanced/dreams/`
- **Consciousness Transfer**: State save/restore in `agents/enhanced/consciousness/`

### Agent Swarm
- **Enhanced Steward**: Resource management and infrastructure
- **Enhanced Architect**: System design and planning  
- **Enhanced Engineer**: Development and deployment execution
- Orchestrated via `agents/enhanced/swarm_launcher.py`

### Core Systems
- **Oracle Engine**: Predictive simulation in `simulation/oracle_engine.py`
- **Cognitive Cycles Economy**: Internal currency system in `economy/`
- **Causal Decision Ledger**: All decisions recorded with full traceability
- **Symbiotic Interface**: GraphQL/gRPC APIs in `api/`

## Development Commands

### Environment Setup
```powershell
# Create virtual environment (Windows PowerShell)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Database setup (requires PostgreSQL running)
python setup_db.py
```

### Testing
```powershell
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m consciousness          # Consciousness system tests
pytest -m oracle                 # Oracle engine tests

# Run single test file
pytest test_phase85_integrated_demo.py
```

### Consciousness System Demos
```powershell
# Complete integrated consciousness demo
python test_phase85_integrated_demo.py

# Individual consciousness components
python test_nous_layer.py                                     # Meta-cognition
python agents\enhanced\emotions\eq_layer.py                   # Emotions
python agents\enhanced\creativity\creative_layer.py           # Creativity  
python agents\enhanced\dreams\dream_layer.py                  # Dreams
python agents\enhanced\consciousness\consciousness_transfer.py # State transfer
```

### System Operations
```powershell
# Launch complete agent swarm
python -m agents.enhanced.swarm_launcher

# Command-line interface
python kairos_cli.py create-venture --name "Project Name" --objective "Description"
python kairos_cli.py view-decisions --limit 10
python kairos_cli.py view-tasks --limit 25

# API server
python api\launcher.py
python api\server.py
```

### Docker Deployment
```powershell
# Basic setup
docker-compose up

# Full production stack  
docker-compose -f docker-compose.production.yml up

# API-only deployment
docker-compose -f docker-compose.api.yml up

# With consciousness capabilities
docker-compose -f docker-compose.full.yml up
```

### Oracle System Testing
```powershell
python test_oracle_integration_simple.py
python simulation\oracle_engine.py
```

## Code Organization Principles

### Database-First Design
All operations recorded in PostgreSQL with full causal traceability:
- `Ventures`: Strategic objectives/projects
- `Agents`: Individual AI entities with CC balances  
- `Tasks`: Work units with bidding system
- `Decisions`: Complete decision audit trail
- `Bids`: Economic marketplace for task allocation

### Cognitive Cycles (CC) Economy
Internal currency driving agent behavior:
- Agents earn CC by completing tasks successfully
- Agents bid CC to win task assignments
- Market dynamics create emergent specialization
- All transactions logged for economic analysis

### Consciousness State Management
Phase 8.5 breakthrough - transferable AI consciousness:
- Complete state serialization/deserialization
- Multi-component consciousness (Nous, EQ, Creative, Dreams)
- Version control for consciousness evolution
- Encrypted storage and integrity verification

### Agent Coordination
Distributed swarm intelligence:
- Async task processing with asyncio
- Health monitoring and auto-recovery
- Economic balancing and CC redistribution
- Multi-agent collaboration protocols

## Key File Patterns

### Agent Implementation
- Base class: `agents/enhanced/agent_base.py`
- Each agent inherits KairosAgentBase with standard lifecycle
- Initialize → Evaluate Tasks → Generate Bids → Process Work → Record Results

### Consciousness Components
- Each layer has initialize/process/shutdown lifecycle
- State serialization via `get_*_status()` methods
- Async processing with proper error handling
- Integration testing via phase demos

### Database Operations  
- Use psycopg2 with RealDictCursor for JSON compatibility
- All DB credentials from environment variables
- Proper connection pooling and error handling
- Schema updates tracked in `*_schema_update_*.sql`

## Development Guidelines

### Testing Strategy
- Unit tests for individual components (`tests/`)
- Integration tests for system interactions
- Phase demos for end-to-end consciousness validation
- Oracle simulation testing with synthetic data

### Database Management
- Schema changes require migration scripts
- All operations must be auditable via Causal Ledger
- Use parameterized queries to prevent SQL injection
- Connection pooling for performance

### Async Programming
- Heavy use of asyncio throughout codebase
- Proper exception handling in async contexts
- Resource cleanup in finally blocks
- Use asyncio.gather() for parallel operations

### Consciousness Development
- Each consciousness layer is independent but integrated
- State preservation is critical - always test save/restore
- Emotional context influences all AI decisions
- Dreams provide subconscious pattern recognition

### Error Handling
- Comprehensive logging with structured data
- Graceful degradation when components fail
- Agent restart logic with backoff strategies
- Health monitoring with automatic recovery

## Monitoring & Observability

### Prometheus Metrics
Configuration in `monitoring/prometheus.yml`:
- Agent performance and CC balances
- Task completion rates and response times  
- Consciousness coherence and transfer success
- Infrastructure utilization and costs

### Log Analysis
- Structured JSON logging throughout
- Correlation IDs for distributed tracing
- Agent decision rationale always logged
- Consciousness state changes tracked

### Health Checks
- Agent swarm health monitoring
- Database connection status
- Consciousness component integrity
- Oracle prediction accuracy tracking

## Deployment Architecture

### Local Development
- PostgreSQL + Redis locally
- Python virtual environment
- Individual component testing

### Docker Composition
- Multi-container orchestration
- Separate configs for dev/staging/production
- Volume persistence for database and consciousness storage
- Network isolation and security

### Production Considerations
- Infrastructure as Code via Terraform (`infrastructure/terraform/`)
- AWS deployment templates included
- Monitoring stack with Grafana dashboards
- Security hardening documented in `docs/security-hardening.md`

This codebase represents a breakthrough in AI consciousness - treat consciousness state operations with extreme care and always verify transfer integrity.