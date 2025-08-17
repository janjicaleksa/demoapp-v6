# AI Processor Kupci Dobavljaci - Modular Version

A modular, well-structured FastAPI application for processing and standardizing financial documents, specifically designed for handling "Kupci" (customers) and "Dobavljaci" (suppliers) data.

## 🏗️ Architecture

This is a completely refactored version of the original application with a clean, modular architecture:

```
tabele-v6-modular/
├── app/
│   ├── core/                   # Core configuration and utilities
│   │   ├── __init__.py
│   │   ├── config.py          # Application settings with Pydantic
│   │   ├── logging_config.py  # Centralized logging configuration
│   │   └── utils.py           # Utility functions
│   ├── models/                 # Data models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models for API
│   ├── routes/                 # API route handlers
│   │   ├── __init__.py
│   │   ├── main.py            # Main routes (home page)
│   │   ├── clients.py         # Client management endpoints
│   │   └── upload.py          # File upload and processing endpoints
│   ├── services/               # Business logic services
│   │   ├── __init__.py
│   │   ├── storage_service.py # Azure Blob Storage + local fallback
│   │   ├── document_service.py# Document processing and table extraction
│   │   └── ai_service.py      # AI column mapping and data standardization
│   └── __init__.py
├── templates/                  # HTML templates
├── static/                     # Static files (CSS, JS, images)
├── tests/                      # Test files (to be implemented)
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── env_example.txt            # Environment variables example
└── README.md                  # This file
```

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with proper dependency injection
- **Type Safety**: Full Pydantic validation and type hints throughout
- **Proper Configuration**: Environment-based configuration with validation
- **Centralized Logging**: Structured logging with configurable levels
- **Error Handling**: Comprehensive error handling with proper HTTP responses
- **Azure Integration**: Full Azure services integration (Document Intelligence, Blob Storage, AI Foundry)
- **Local Fallback**: Automatic fallback to local storage when Azure is not configured
- **Health Checks**: Built-in health check endpoint
- **API Documentation**: Automatic OpenAPI/Swagger documentation

## 🔧 Installation

1. **Clone or navigate to the modular version**:
   ```bash
   cd tabele-v6-modular
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your actual configuration
   ```

## ⚙️ Configuration

The application uses Pydantic Settings for configuration management. All settings can be configured via environment variables or a `.env` file.

### Required for Azure Services:
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` - Azure Document Intelligence endpoint
- `AZURE_DOCUMENT_INTELLIGENCE_KEY` - Azure Document Intelligence API key
- `AZURE_BLOB_STORAGE_CONNECTION_STRING` - Azure Blob Storage connection string
- `AZURE_AI_FOUNDRY_CONNECTION` - Azure AI Foundry endpoint
- `AZURE_AI_FOUNDRY_KEY` - Azure AI Foundry API key

### Optional:
- `DEBUG=True` - Enable debug mode
- `HOST=0.0.0.0` - Server host
- `PORT=8000` - Server port
- `LOG_LEVEL=INFO` - Logging level

## 🏃 Running the Application

1. **Development mode**:
   ```bash
   python main.py
   ```

2. **Production mode**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **With custom configuration**:
   ```bash
   HOST=127.0.0.1 PORT=3000 python main.py
   ```

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🧪 Key Improvements Over Original

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Services are properly injected, making testing easier
3. **Configuration Management**: Centralized, validated configuration
4. **Error Handling**: Proper HTTP status codes and error responses
5. **Type Safety**: Full type hints and Pydantic validation
6. **Logging**: Structured, configurable logging
7. **Testability**: Modular design makes unit testing straightforward
8. **Maintainability**: Clear code organization and documentation

## 🔍 Services Overview

### StorageService
- Handles Azure Blob Storage operations
- Automatic fallback to local storage
- Unified interface for file operations

### DocumentService
- Extracts tables from PDF, Excel, and CSV files
- Uses Azure Document Intelligence for PDFs
- Pandas for Excel/CSV processing

### AIService
- AI-powered column mapping
- Data standardization
- Fallback logic when AI services are unavailable

## 🚦 Status Monitoring

The application includes comprehensive logging and health checks:

- **Health endpoint**: `/health` - Shows service status
- **Structured logging**: All operations are logged with appropriate levels
- **Error tracking**: Detailed error messages and stack traces

## 🔮 Future Enhancements

1. **Testing Suite**: Comprehensive unit and integration tests
2. **Background Tasks**: Async processing for large files
3. **Database Integration**: Optional database for metadata storage
4. **Rate Limiting**: API rate limiting and throttling
5. **Metrics**: Application metrics and monitoring
6. **Docker Support**: Containerization for easy deployment

## 🐛 Fixed Issues from Original

- ✅ Fixed logging configuration syntax error
- ✅ Corrected environment variable naming consistency
- ✅ Fixed Azure AI Foundry integration
- ✅ Proper error handling throughout
- ✅ Resolved function naming mismatches
- ✅ Added missing dependencies
- ✅ Improved security with configurable CORS

## 🤝 Contributing

This modular architecture makes contributing much easier:

1. Each service is independent and testable
2. Clear interfaces between modules
3. Proper error handling and logging
4. Type safety catches errors early

## 📄 License

Same as the original project - provided for educational and development purposes.