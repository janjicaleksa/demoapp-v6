# Easy Tables - Simplified AI Processor

A streamlined version of the AI Processor Kupci Dobavljaci application, focused on parameter-based organization instead of date-based structure.

## Key Features

- **Parameter-Based Organization**: Files are organized by `file_type` (kupci/dobavljaci) and `period` (prethodna_fiskalna_godina/presek_bilansa_tekuca_godina) instead of manual date entry
- **Simplified Upload**: Choose file type and period from dropdowns - no manual date input required
- **Azure AI Document Intelligence**: Automatic text extraction from documents
- **Custom AI Processing**: Uses your specific prompt for data extraction and standardization
- **Two Upload Modes**: Single file or batch upload
- **IV Support**: Optional Ispravka Vrednosti flag for kupci files

## Directory Structure

Files are automatically organized as:
```
clients/
└── [client_slug]/
    ├── kupci/
    │   ├── prethodna_fiskalna_godina/
    │   │   ├── raw_[filename]
    │   │   ├── extracted_[filename].txt
    │   │   ├── ai_result_[filename].json
    │   │   └── iv/  # For IV files
    │   └── presek_bilansa_tekuca_godina/
    │       ├── raw_[filename]
    │       ├── extracted_[filename].txt
    │       ├── ai_result_[filename].json
    │       └── iv/  # For IV files
    └── dobavljaci/
        ├── prethodna_fiskalna_godina/
        │   ├── raw_[filename]
        │   ├── extracted_[filename].txt
        │   └── ai_result_[filename].json
        └── presek_bilansa_tekuca_godina/
            ├── raw_[filename]
            ├── extracted_[filename].txt
            └── ai_result_[filename].json
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   - Copy `env_example.txt` to `.env`
   - Fill in your Azure credentials:
     - Azure Document Intelligence endpoint and key
     - Azure AI Foundry connection and key

3. **Run Application**:
   ```bash
   python start.py
   ```
   Or:
   ```bash
   python main.py
   ```

4. **Access Application**:
   Open http://localhost:8000 in your browser

## Usage

### Step 1: Create Client
Enter a client name to create the parameter-based directory structure.

### Step 2: Choose Upload Mode

#### Single File Mode
- Select file type: "kupci" or "dobavljaci"
- Select period: "Prethodna Fiskalna Godina" or "Presek Bilansa Tekuća Godina"
- For kupci files: optionally check "Ispravka Vrednosti (IV)"
- Upload and process one file at a time

#### Batch Mode
Upload multiple files simultaneously:
- **Prethodna Fiskalna Godina**: kupci, dobavljaci, kupci IV (optional)
- **Presek Bilansa Tekuća Godina**: kupci, dobavljaci, kupci IV (optional)

## AI Processing

The application uses your custom prompt to extract and standardize financial data:

### Input Parameters
- `file_type`: "kupci" or "dobavljaci"
- `period_date`: "kraj prethodne fiskalne godine" or "presek bilansa tekuće godine"
- `desired_headers`: Automatically set based on file type
- `is_ispravka_vrednosti_kupci`: Boolean flag for IV files

### Output Format
```json
{
  "file_type": "kupci",
  "period_date": "kraj prethodne fiskalne godine",
  "datum": "2024-12-31",
  "ispravka_vrednosti": false,
  "tabela": [
    {
      "konto": "120",
      "naziv partnera": "Company Name",
      "promet duguje": 1000.00,
      "saldo": 1000.00
    }
  ]
}
```

## API Endpoints

- `POST /api/clients`: Create a new client
- `POST /api/upload/process`: Process a single file
- `POST /api/upload/batch`: Process multiple files in batch
- `GET /`: Main application interface

## File Support

Supported file formats:
- Excel (.xlsx)
- CSV (.csv)
- Word (.doc, .docx)
- PDF (.pdf)

Maximum file size: 50MB

## Differences from Original

1. **No Manual Date Entry**: Periods are selected from predefined options
2. **Parameter-Based Structure**: Organization by file type and period instead of dates
3. **Simplified Interface**: Streamlined UI focused on file type and period selection
4. **Custom AI Prompt**: Uses your specific prompt for data extraction
5. **Automatic Headers**: Desired headers are automatically determined based on file type

## Configuration

All settings are managed through environment variables. See `env_example.txt` for all available options.

Key configurations:
- Azure Document Intelligence for text extraction
- Azure AI Foundry for custom prompt processing
- File size limits and allowed extensions
- Storage paths and organization