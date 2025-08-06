# AI Processor Kupci Dobavljaci

A FastAPI-based web application for processing and standardizing financial documents, specifically designed for handling "Kupci" (customers) and "Dobavljaci" (suppliers) data.

## Features

- **Client Management**: Create and manage clients with normalized folder structures
- **Document Processing**: Support for Excel, CSV, PDF, and Word documents
- **AI-Powered Standardization**: Convert various table formats to standardized JSON structure
- **Azure Document Intelligence Integration**: PDF table extraction using Azure AI
- **Azure Blob Storage Integration**: Cloud-based file storage with local fallback
- **Period-based Organization**: Automatic folder creation based on period dates
- **Modern UI**: Clean, responsive interface built with Tailwind CSS

## Project Structure

```
tabele-v6/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Main web interface
├── static/                # Static files (CSS, JS, images)
└── clients/               # Generated client data (local fallback)
    └── {client_slug}/
        ├── raw/
        │   └── {period_date}/
        └── processed/
            └── {period_date}/
```

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Azure Document Intelligence** (optional, for PDF processing):
   - Create an Azure account and set up Document Intelligence service
   - Set environment variables:
     ```bash
     export AZURE_ENDPOINT="your-azure-endpoint"
     export AZURE_KEY="your-azure-key"
     ```
   - Or create a `.env` file:
     ```
     AZURE_ENDPOINT=your-azure-endpoint
     AZURE_KEY=your-azure-key
     ```

4. **Set up Azure Blob Storage** (optional, for cloud storage):
   - Create an Azure Storage Account
   - Get the connection string from Azure Portal > Storage Account > Access keys
   - Set environment variable:
     ```bash
     export AZURE_STORAGE_CONNECTION_STRING="your-connection-string"
     ```
   - Or add to `.env` file:
     ```
     AZURE_STORAGE_CONNECTION_STRING=your-connection-string
     ```

## Storage Configuration

The application supports two storage modes:

### Azure Blob Storage (Recommended)
- **Automatic**: When `AZURE_STORAGE_CONNECTION_STRING` is configured
- **Benefits**: Scalable, reliable, cloud-based storage
- **Structure**: Virtual folders in Azure Blob Storage container
- **Path format**: `azure://clients/{client_slug}/{raw|processed}/{period_date}/`

### Local Storage (Fallback)
- **Automatic**: When Azure Blob Storage is not configured
- **Benefits**: Simple, no external dependencies
- **Structure**: Local file system directories
- **Path format**: `clients/{client_slug}/{raw|processed}/{period_date}/`

## Usage

1. **Start the application**:
   ```bash
   python main.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:8000`

3. **Workflow**:
   - **Step 1**: Enter client name (automatically normalized to create folder structure)
   - **Step 2**: Set period date and choose upload mode
   - **Step 3**: Upload files based on selected mode

## Upload Modes

### Fixed Upload (Summary Reports)
Required files:
- `kupci-kraj-fiskalne-godine` (Customers end of fiscal year)
- `kupci-bilans-preseka` (Customers balance sheet)
- `dobavljaci-kraj-fiskalne-godine` (Suppliers end of fiscal year)
- `dobavljaci-bilans-preseka` (Suppliers balance sheet)

Optional files:
- `kupci-kraj-fiskalne-godine-iv`
- `kupci-bilans-preseka-iv`

### Unfixed Upload (Per Account)
- Currently a placeholder for future implementation
- Files are saved but not processed

## File Processing

### Supported Formats
- **Excel (.xlsx)**: Direct table extraction from all sheets
- **CSV (.csv)**: Direct table parsing
- **PDF (.pdf)**: Azure Document Intelligence for table extraction
- **Word (.doc/.docx)**: Basic support (saved but not processed)

### Standardization
The AI processor converts various table formats into a standardized JSON structure with columns:
- `konto` (Account)
- `naziv_partnera` (Partner Name)
- `duguje` (Debit)
- `potrazuje` (Credit)
- `saldo` (Balance)

## API Endpoints

- `GET /`: Main web interface
- `POST /api/clients`: Create new client
- `POST /api/upload/fixed`: Upload fixed files for processing
- `POST /api/upload/unfixed`: Upload unfixed files (placeholder)
- `GET /api/clients/{client_slug}/structure`: Get client folder structure

## Folder Structure

When a client is created, the following structure is automatically generated:

### Azure Blob Storage
```
azure://clients/{client_slug}/
├── raw/
│   └── {period_date}/
│       ├── kupci-kraj-fiskalne-godine.xlsx
│       ├── kupci-bilans-preseka.pdf
│       └── ...
└── processed/
    └── {period_date}/
        ├── kupci-kraj-fiskalne-godine-processed.json
        ├── kupci-bilans-preseka-processed.json
        └── ...
```

### Local Storage
```
clients/
└── {client_slug}/
    ├── raw/
    │   └── {period_date}/
    │       ├── kupci-kraj-fiskalne-godine.xlsx
    │       ├── kupci-bilans-preseka.pdf
    │       └── ...
    └── processed/
        └── {period_date}/
            ├── kupci-kraj-fiskalne-godine-processed.json
            ├── kupci-bilans-preseka-processed.json
            └── ...
```

## Configuration

### Environment Variables
- `AZURE_ENDPOINT`: Azure Document Intelligence endpoint
- `AZURE_KEY`: Azure Document Intelligence API key
- `AZURE_STORAGE_CONNECTION_STRING`: Azure Blob Storage connection string (optional)

### File Size Limits
- Default FastAPI file upload limits apply
- Large files may need server configuration adjustments

## Development

### Adding New File Types
1. Add the file extension to `allowed_extensions` in the upload endpoint
2. Create a new extraction function (similar to `extract_tables_from_excel`)
3. Add the extraction logic to the file processing section

### Enhancing AI Standardization
The current `standardize_table_data` function uses simple pattern matching. For production use, consider:
- Implementing a proper AI model (e.g., using Azure Cognitive Services)
- Adding more sophisticated data validation
- Implementing machine learning for better pattern recognition

## Troubleshooting

### Common Issues

1. **Azure credentials not configured**:
   - PDF processing will be skipped
   - Check environment variables or `.env` file

2. **Azure Blob Storage not configured**:
   - Application will automatically fall back to local storage
   - Check `AZURE_STORAGE_CONNECTION_STRING` environment variable

3. **File upload errors**:
   - Ensure file types are supported
   - Check file size limits
   - Verify file integrity

4. **Processing errors**:
   - Check file format compatibility
   - Verify Azure service status (for PDFs)
   - Review server logs for detailed error messages

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests! 