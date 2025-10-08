# Insurance PDF Comparison App (Landing AI + Flask)

AI-powered web app to compare Mexican auto insurance quotations from multiple PDFs. Uses Landing AI Agentic Document Extraction with graceful fallbacks and professional export.

## Stack
- Backend: Flask (WSGI) + ASGI wrapper via `asgiref` (served by `uvicorn`)
- AI: `agentic-doc` (Landing AI)
- PDF text fallback: PyMuPDF (`fitz`)
- Export: WeasyPrint (with HTML fallback)
- Frontend: HTML, CSS, JavaScript (vanilla)
- Deployment: Railway (`Procfile`, `Dockerfile`)

## Supported Companies
- HDI Seguros
- Qualitas
- ANA Seguros
- Seguros Atlas

## Features
- Landing AI classification + company-specific extraction
- Fallback to text/regex parsing if AI unavailable
- Standardized comparison table and automatic IVA (16%)
- Editable cells and live recalculation (client-side)
- Export to PDF (WeasyPrint) or HTML fallback
- Simple caching, health endpoint, logging

## Setup
1. Python 3.13 recommended (see `.python-version`).
2. Install dependencies:
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
   - macOS/Linux:
     ```bash
     python -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt
     ```
3. Create `.env` with:
   ```env
   LANDING_AI_API_KEY=your_api_key_here
   FLASK_SECRET_KEY=change-me
   PORT=8000
   ```
   Note: The app auto-loads `.env` via `python-dotenv` if available.

## Run
- WSGI (Flask dev server):
  ```bash
  python app.py
  ```
- ASGI (recommended):
  ```bash
  uvicorn app:asgi_app --host 0.0.0.0 --port 8000
  ```
Open `http://localhost:8000` and upload PDFs.

## Export Notes
- WeasyPrint requires system libraries (cairo/pango). The included `Dockerfile` installs them.
- On Windows without WeasyPrint, export route returns HTML you can print to PDF.

## Project Structure
- `app.py`: routes, AI parsing, fallback, business logic, export
- `templates/`: `index.html`, `results.html`, `export.html`
- `static/style.css`: UI + print styles
- `brands.json`: aids vehicle extraction
- Deployment: `Procfile`, `Dockerfile`, `requirements.txt`

## Deployment (Railway)
- Set `LANDING_AI_API_KEY`, `FLASK_SECRET_KEY` (Railway sets `PORT`).
- `Procfile` runs `uvicorn app:asgi_app`.
- See `DEPLOYMENT.md` for details.

## Logos
- Placeholder SVG data-URIs are used by default. Replace with real PNG logos if desired and update logic in `app.py` accordingly.

## Testing Tips
- Try multiple PDFs from supported insurers.
- Validate IVA/total updates after editing numeric fields in results.

License: MIT
