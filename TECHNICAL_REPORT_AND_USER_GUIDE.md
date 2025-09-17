# Technical Report and User Guide

## Executive Summary

This document serves as both a technical report and user guide for the AI-Powered Financial Intelligence System. The system integrates financial data from QuickBooks and Rootfi, providing natural language query capabilities, automated insights, and predictive analytics through a multi-model AI architecture.

---

# Part 1: Technical Report

## System Architecture

### Core Components

1. **FastAPI Backend** - Asynchronous Python web framework for high-performance API endpoints
2. **SQLAlchemy + SQLite** - ORM with async support for data persistence
3. **LangChain Framework** - LLM orchestration for query processing and response generation
4. **Multi-Model AI System** - Intelligent routing between GPT-5, Claude Sonnet, and Claude Opus
5. **Analytics Engine** - ML-based forecasting and statistical anomaly detection

### Data Flow Architecture

```
User Query → API Gateway → Query Processor → LLM Router → Model Selection
    ↓                                             ↓
Response ← Response Formatter ← SQL Executor ← SQL Generator
```

## Design Decisions

### 1. Multi-Model LLM Architecture

**Decision**: Implement dynamic model routing based on query complexity

**Rationale**:

- Cost optimization: Simple queries use cheaper models (GPT-5)
- Quality optimization: Complex queries use more capable models (Claude Sonnet)
- Verification: Critical queries can use Claude Opus for validation

**Implementation**:

- Complexity assessment using heuristics (keywords, query length, operations)
- Configurable model mapping in settings
- Fallback mechanisms for model failures

### 2. Natural Language to SQL Pipeline

**Decision**: Use LangChain with structured output parsing

**Rationale**:

- Type safety through Pydantic models
- Consistent query structure
- SQL injection prevention
- Context-aware query generation

**Components**:

- ParsedQuery: Structured representation of user intent
- SQLQuery: Safe SQL generation with parameterization
- Time period parsing with multiple format support

### 3. Context Management

**Decision**: In-memory session storage with LangChain conversation buffers

**Rationale**:

- Enables follow-up questions
- Maintains entity tracking
- Summarizes long conversations
- Fast access for real-time responses

**Limitations**:

- Sessions lost on restart
- Not suitable for horizontal scaling without Redis

### 4. Analytics Integration

**Decision**: Separate analytics tools invoked by AI

**Rationale**:

- Clear separation of concerns
- AI determines when to use analytics
- Easier to test and maintain
- Extensible for future tools

## Performance Considerations

### Query Processing

- Async/await throughout for non-blocking I/O
- Connection pooling for database
- Streaming responses for large results
- Model-specific optimizations (token limits, temperature)

### Caching Strategy

- LLM responses not cached (context-dependent)
- SQL query results cacheable (not implemented)
- Model instances reused across requests

### Scalability

- Vertical scaling: Increase server resources
- Horizontal scaling: Requires session storage migration to Redis
- Database: SQLite suitable for POC, migrate to PostgreSQL for production

## Security Measures

### SQL Injection Prevention

1. Parameterized queries throughout
2. Keyword blocking for dangerous SQL operations
3. Query validation before execution
4. Limited query scope to SELECT operations

### API Security

- Input validation on all endpoints
- Rate limiting ready (not implemented)
- API key management through environment variables
- CORS configuration for frontend integration

### Data Privacy

- No PII logging
- Configurable data retention
- Session isolation
- Audit trail capability (not implemented)

---

# Part 2: User Guide

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for OpenAI and Anthropic
- 2GB+ RAM recommended
- Git for cloning repository

### Quick Start

1. **Clone and Setup**

```bash
git clone <repository-url>
cd kudwa_assessment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

2. **Configure API Keys**
   Create `keys.env`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

3. **Initialize and Run**

```bash
python -c "from main import init_db; import asyncio; asyncio.run(init_db())"
python main.py
```

4. **Import Data**

```bash
curl -X POST http://localhost:8000/api/data/import/quickbooks -H "Content-Type: application/json" -d @data_set_1.json
curl -X POST http://localhost:8000/api/data/import/rootfi -H "Content-Type: application/json" -d @data_set_2.json
```

## Using the System

### Natural Language Queries

The system understands various query types:

**Basic Queries**

- "What was total revenue last quarter?"
- "Show me expenses for March 2024"
- "What's our current cash position?"

**Comparisons**

- "Compare Q1 and Q2 revenue"
- "How did expenses change year over year?"

**Calculations**

- "Calculate profit margin for 2024"
- "What's the expense ratio by category?"

**Forecasting**

- "Forecast revenue for next 3 months"
- "Predict Q2 2025 expenses"

**Anomaly Detection**

- "Show unusual expense patterns"
- "Are there any revenue anomalies?"

### API Reference

#### Core Endpoints

**Natural Language Query**

```http
POST /api/queries/natural
Content-Type: application/json

{
  "query": "Your question here",
  "session_id": "optional-session-id"
}
```

**Streaming Query**

```http
POST /api/queries/natural/stream
Content-Type: application/json

{
  "query": "Your question here",
  "session_id": "optional-session-id"
}
```

**Financial Data Access**

```http
GET /api/financial/records?start_date=2024-01-01&end_date=2024-12-31
GET /api/financial/summary?company_id=1
GET /api/financial/trends?metric=revenue&period=monthly
```

### Common Use Cases

#### Financial Reporting

```python
# Monthly revenue report
query = "Show me monthly revenue breakdown for 2024"

# Expense analysis
query = "What are our top 5 expense categories this quarter?"

# Profit trends
query = "How has our profit margin changed over the last 6 months?"
```

#### Forecasting

```python
# Revenue projection
query = "Forecast revenue for Q2 2025 based on current trends"

# Budget planning
query = "Predict expenses for next quarter if we maintain current spending"
```

#### Anomaly Detection

```python
# Expense monitoring
query = "Alert me to any unusual expense patterns this month"

# Revenue validation
query = "Check for any anomalous revenue entries in Q1"
```


### Debug Mode

Enable detailed logging:

```python
# In main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# System health
curl http://localhost:8000/health

# AI service health
curl http://localhost:8000/api/queries/health
```

## Model Selection Guide

### When Each Model is Used

**GPT-5 (Default)**

- Simple lookups and aggregations
- Basic time period queries
- Single metric requests

**Claude Sonnet**

- Multi-step analyses
- Complex comparisons
- Forecasting requests

**Claude Opus**

- Verification tasks
- High-stakes queries
- Quality assurance

### Forcing Model Selection

```json
{
  "query": "Your question",
  "model_preference": "claude-sonnet-4-20250514"
}
```

## Limitations

1. **Data Scope**

   - Only processes imported financial data
   - No real-time data integration
   - Limited to structured financial metrics

2. **Analytical Capabilities**

   - Forecasting: Simple ML models, not deep learning
   - Anomaly detection: Statistical methods only
   - No custom metric definitions

3. **Technical Constraints**

   - Session state in memory (resets on restart)
   - SQLite database (single-user optimal)
   - No horizontal scaling without modifications

4. **Query Limitations**
   - No data modification queries
   - Limited to SELECT operations
   - No cross-company comparisons in single query
