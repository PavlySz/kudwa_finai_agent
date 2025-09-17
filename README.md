# AI-Powered Financial Intelligence System

An intelligent financial data processing system that enables natural language queries, automated insights, and predictive analytics for financial data.

## Project Overview

This system integrates diverse financial data sources (QuickBooks and Rootfi) into a unified backend with powerful AI capabilities. Users can query financial data using natural language, receive contextual insights, and leverage advanced analytics like forecasting and anomaly detection.

### Key Features

- **Natural Language Queries**: Ask questions in plain English about your financial data
- **Multi-Model AI Routing**: Intelligently routes queries to GPT-5, Claude Sonnet, or Claude Opus based on complexity
- **Contextual Conversations**: Maintains conversation history for follow-up questions
- **Financial Forecasting**: Predict future revenue and expenses using ML models
- **Anomaly Detection**: Automatically identify unusual patterns in financial data
- **RESTful APIs**: Direct programmatic access to all financial data
- **Comprehensive Testing**: Automated evaluation framework to ensure AI quality

## Technical Architecture

### System Components

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   FastAPI App   │ ───> |  AI Processing   │ ───> │   Analytics     │
│   (main.py)     │      │  - LLM Client    │      │  - Forecasting  │
│                 │      │  - Query Parser  │      │  - Anomaly Det. │
└─────────────────┘      │  - SQL Generator │      └─────────────────┘
         │               └──────────────────┘                │
         ▼                       │                           │
┌─────────────────┐              ▼                           ▼
│  Data Storage   │      ┌──────────────────┐      ┌─────────────────┐
│  - SQLite DB    │ ───> │ Context Manager  │ ───> │ Response Format │
│  - Models       │      │ - Session State  │      │ - Narratives    │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

### Tech Stack

- **Backend**: FastAPI (async Python web framework)
- **Database**: SQLAlchemy + SQLite (easily upgradeable to PostgreSQL)
- **AI/LLM**: LangChain + OpenAI GPT-5 + Anthropic Claude
- **Analytics**: scikit-learn + pandas + numpy
- **Testing**: pytest + custom evaluation framework

## Features

### 1. Natural Language Queries

Query your financial data conversationally:

- "What was our revenue last quarter?"
- "Show me expense trends for 2024"
- "Compare Q1 and Q2 profit margins"
- "Which categories had the highest spending?"

### 2. AI-Powered Analytics

- **Forecasting**: "Predict next quarter's revenue" - Uses ML models (XGBoost + moving averages)
- **Anomaly Detection**: "Show me any unusual expense patterns" - Statistical anomaly detection with z-scores
- **Trend Analysis**: "How are our margins trending?" - Identifies patterns and directions
- **Comparisons**: "Compare this year vs last year performance" - Side-by-side analysis

**Supported Analytics Queries:**

- "Forecast revenue for next 3 months"
- "Predict expenses for Q2 2024"
- "Are there anomalies in our financial data?"
- "Show unusual patterns in expenses"
- "What will our cash flow look like next quarter?"

### 3. Multi-Model Intelligence

The system automatically routes queries to the most appropriate AI model:

- **GPT-5**: Default for general queries
- **Claude Sonnet**: Complex multi-step analysis
- **Claude Opus**: Verification and high-accuracy tasks

### 4. RESTful API Endpoints

#### AI Query Endpoints

- `POST /api/queries/natural` - Natural language queries
- `POST /api/queries/natural/stream` - Streaming responses
- `GET /api/queries/sessions/{id}` - Session history
- `POST /api/queries/feedback` - Submit feedback

#### Financial Data APIs

- `GET /api/financial/records` - Retrieve financial records
- `GET /api/financial/summary` - Financial summaries
- `GET /api/financial/trends` - Trend analysis
- `GET /api/financial/categories` - Category breakdowns

#### Evaluation APIs

- `POST /api/eval/test-suite/run` - Run test suite
- `POST /api/eval/judge` - LLM quality evaluation
- `GET /api/eval/metrics/dashboard` - Performance metrics

## Setup Instructions

### Prerequisites

- Python 3.11+
- pip
- API keys for OpenAI and Anthropic

### 1. Clone the Repository

```bash
git clone <repository-url>
cd kudwa_assessment
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `keys.env` file in the root directory:

```env
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### 5. Configure Deployment Mode

The system can run locally or on Replit. In `src/config/settings.py`, set:

```python
USE_REPLIT = False  # Set to True when deploying on Replit
```

This automatically configures the API URL:

- Local: `http://localhost:8000`
- Replit: `https://pavly-kudwa-finai-agent.replit.app`

All test scripts will use the configured URL automatically.

### 6. Initialize Database

```bash
python -c "from main import init_db; import asyncio; asyncio.run(init_db())"
```

### 7. Import Sample Data

```bash
# Import QuickBooks data
curl -X POST http://localhost:8000/api/data/import/quickbooks \
  -H "Content-Type: application/json" \
  -d @data_set_1.json

# Import Rootfi data
curl -X POST http://localhost:8000/api/data/import/rootfi \
  -H "Content-Type: application/json" \
  -d @data_set_2.json
```

### 8. Run the Server

```bash
python main.py
```

The server will start at the configured URL (local or Replit based on settings)

## Example Usage

### Natural Language Query

```bash
curl -X POST http://localhost:8000/api/queries/natural \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was our total revenue in Q1 2024?",
    "session_id": "user-123"
  }'
```

### Response

```json
{
  "success": true,
  "answer": "Your total revenue in Q1 2024 was $487,234. This represents a 15% increase compared to Q4 2023.",
  "key_insights": [
    "March was the strongest month with $180,432 in revenue",
    "Service revenue grew 22% while product revenue remained flat"
  ],
  "metadata": {
    "intent": "aggregation",
    "metrics": ["revenue"],
    "time_period": "Q1 2024"
  }
}
```

### Forecasting Query

```bash
curl -X POST http://localhost:8000/api/queries/natural \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Forecast our revenue for the next 3 months",
    "session_id": "user-123"
  }'
```

### Anomaly Detection

```bash
curl -X POST http://localhost:8000/api/queries/natural \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Are there any unusual patterns in our expenses?",
    "session_id": "user-123"
  }'
```

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run Evaluation Suite

```bash
python tests/test_evaluation_api.py
```

### Test Chat Interface

The chat test script automatically uses the configured API URL:

```bash
# Will use localhost:8000 or Replit URL based on settings
python test_chat.py
```

### Test Analytics Features

```bash
python test_analytics.py
```

### Test Specific Components

```bash
# Test LLM connections
python tests/test_llm_connection_routing.py

# Test query processing
python tests/test_query_processor.py

# Test financial APIs
python tests/test_financial_api.py
```

## Deployment

### For Replit

1. Upload the project to Replit
2. Set environment variables in Replit Secrets
3. Run `python main.py`
4. Access via the Replit URL

### For Local Development

Simply run:

```bash
python main.py
```

Access the API at:

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Performance & Limitations

### Current Capabilities

- Handles queries about revenue, expenses, profit, and cash flow
- Supports time-based analysis (daily, monthly, quarterly, yearly)
- Processes follow-up questions with context awareness
- Provides forecasts up to 12 months ahead
- Detects statistical anomalies using z-score method

### Known Limitations

- Forecasting requires at least 3 historical data points
- Anomaly detection is based on statistical methods (not deep learning)
- Complex SQL joins may need optimization for large datasets
- Session context is stored in memory (resets on restart)

## Project Structure

```
kudwa_assessment/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── keys.env               # API keys (create this)
├── README.md              # This file
├── .replit                # Replit configuration
├── replit.nix            # Replit dependencies
├── data_set_1.json       # QuickBooks sample data
├── data_set_2.json       # Rootfi sample data
├── financial.db          # SQLite database (auto-created)
│
├── src/
│   ├── config/
│   │   ├── database.py   # Database configuration
│   │   └── settings.py   # Application settings
│   │
│   ├── models/
│   │   ├── base.py       # Base models and enums
│   │   ├── company.py    # Company model
│   │   └── financial.py  # Financial record models
│   │
│   ├── services/
│   │   ├── parsers/      # Data parsers for QB/Rootfi
│   │   └── data_import.py # Data import service
│   │
│   ├── ai/
│   │   ├── llm_client.py      # Multi-model LLM client
│   │   ├── query_processor.py  # NL to SQL conversion
│   │   ├── response_formatter.py # Response generation
│   │   └── context_manager.py   # Conversation state
│   │
│   ├── analytics/
│   │   ├── forecaster.py      # ML forecasting
│   │   └── anomaly_detector.py # Anomaly detection
│   │
│   ├── api/
│   │   ├── health.py      # Health check endpoints
│   │   ├── data_import.py # Data import endpoints
│   │   ├── ai_queries.py  # Natural language endpoints
│   │   ├── financial.py   # Financial data APIs
│   │   └── evaluation.py  # Evaluation endpoints
│   │
│   └── evaluation/
│       ├── test_suite.py  # AI test cases
│       ├── llm_judge.py   # LLM-as-Judge system
│       └── metrics.py     # Performance metrics
│
└── tests/
    ├── test_llm_connection_routing.py
    ├── test_query_processor.py
    ├── test_ai_integration.py
    ├── test_financial_api.py
    └── test_evaluation_api.py
```


## Deployment

### Replit Deployment

The system is currently deployed on Replit at: https://pavly-kudwa-finai-agent.replit.app

To use the Replit deployment:

1. Set `USE_REPLIT = True` in `src/config/settings.py`
2. All test scripts will automatically use the Replit URL
3. API documentation available at: https://pavly-kudwa-finai-agent.replit.app/docs

### Switching Between Local and Replit

The deployment mode is controlled by a single setting:

```python
# src/config/settings.py
USE_REPLIT = False  # Local development
# or
USE_REPLIT = True   # Replit deployment
```

## Next Steps and Future Improvements

### 1. Production Deployment

**Docker & Containerization**

- Multi-stage Dockerfile
- Docker Compose with PostgreSQL, Redis, and monitoring stack
- Kubernetes manifests for cloud deployment

**Infrastructure Improvements**

- Migrate from SQLite to PostgreSQL for concurrent users
- Redis for session storage and caching
- nginx reverse proxy with rate limiting
- Monitoring with Prometheus

### 2. Model & AI Improvements

**Enhanced Model Routing**

- ML-based complexity assessment instead of heuristics
- Dynamic model selection based on query performance history
- A/B testing framework for model comparison
- Cost-benefit optimization algorithm
- Stress-testing the routing

**Better Context Management**

- Vector database for long-term memory (ChromaDB/Pinecone)
- Cross-session learning and insights
- Automatic context summarization

### 3. Advanced Analytics

**Forecasting Enhancements**

- LSTM/Transformer models for time series
- Seasonality detection and adjustment
- External factor integration (market data, holidays)
- Confidence interval refinement
- Multi-variate forecasting

**Anomaly Detection Upgrades**

- Isolation Forest implementation
- LSTM autoencoders for sequence anomalies
- Real-time anomaly alerts
- Contextual anomaly explanations
- Threshold learning per metric

### 4. RAG Implementation

**Retrieval-Augmented Generation**

- Document store for financial reports/policies
- Semantic search over historical insights
- Source citation in responses
- Custom knowledge base integration
- Regulatory compliance database

**Vector Search Integration**

- Embed financial transactions
- Similar pattern matching
- Historical query retrieval
- Insight recommendation engine

### 5. Tool Ecosystem

**Financial Tools**

- Direct QuickBooks API integration
- Real-time data streaming
- Cash flow projections
- Financial ratio calculations

**Integration Tools**

- Slack/Teams notifications
- Email report generation
- Calendar integration for periodic reports
- Export to Excel/Google Sheets

### 6. Business Intelligence Features

**Enhacned Reporting**

- Custom dashboard builder
- Scheduled report generation
- KPI tracking and alerts
- Data visualization library

**Predictive Analytics**

- Customer churn prediction
- Revenue opportunity identification
- Expense optimization suggestions
- Cash flow risk assessment
- Seasonal trend analysis
