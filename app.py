import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    Response,
)
from pydantic import BaseModel, Field
from asgiref.wsgi import WsgiToAsgi


# Optional imports with graceful degradation
try:
    from agentic_doc.parse import parse as landing_parse
    _LANDING_AI_AVAILABLE = True
except Exception:
    landing_parse = None  # type: ignore
    _LANDING_AI_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except Exception:
    fitz = None  # type: ignore
    _PYMUPDF_AVAILABLE = False

try:
    from weasyprint import HTML, CSS
    _WEASYPRINT_AVAILABLE = True
except Exception:
    HTML = None  # type: ignore
    CSS = None  # type: ignore
    _WEASYPRINT_AVAILABLE = False


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("autos-seguros")


# Cache (simple in-memory; replace with Redis for production if needed)
_parse_cache: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Models and Schemas for Landing AI Extraction
# -----------------------------------------------------------------------------


class InsuranceData(BaseModel):
    """Base model for insurance data extraction"""

    company: str = Field(description="Insurance company name")
    vehicle_name: str = Field(description="Vehicle description (brand, model, year)")
    forma_de_pago: str = Field(description="Payment method")
    danos_materiales: float = Field(description="Material damage coverage amount")
    deducible_dm: str = Field(description="Material damage deductible")
    robo_total: float = Field(description="Total theft coverage amount")
    deducible_rt: str = Field(description="Theft deductible")
    responsabilidad_civil: float = Field(description="Civil liability coverage amount")
    gastos_medicos_ocupantes: float = Field(description="Occupant medical expenses coverage")
    asistencia_legal: str = Field(description="Legal assistance coverage")
    asistencia_viajes: str = Field(description="Travel assistance coverage")
    accidente_conductor: float = Field(description="Driver accident coverage")
    responsabilidad_civil_catastrofica: float = Field(
        description="Catastrophic civil liability coverage"
    )
    desbielamiento_agua_motor: str = Field(
        description="Engine water damage coverage"
    )
    prima_neta: float = Field(description="Net premium amount")
    recargos: float = Field(description="Surcharges amount")
    derechos_poliza: float = Field(description="Policy rights amount")
    iva: float = Field(description="Tax amount")
    prima_total: float = Field(description="Total premium amount")


class HDISegurosData(InsuranceData):
    company: str = Field(default="HDI Seguros")


class QualitasData(InsuranceData):
    company: str = Field(default="Qualitas")


class ANASegurosData(InsuranceData):
    company: str = Field(default="ANA Seguros")


class AtlasSegurosData(InsuranceData):
    company: str = Field(default="Seguros Atlas")


classification_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [
                "HDI Seguros",
                "Qualitas",
                "ANA Seguros",
                "Seguros Atlas",
                "Other",
            ],
        }
    },
    "required": ["document_type"],
}


def get_extraction_schema(document_type: str):
    mapping = {
        "HDI Seguros": HDISegurosData,
        "Qualitas": QualitasData,
        "ANA Seguros": ANASegurosData,
        "Seguros Atlas": AtlasSegurosData,
    }
    return mapping.get(document_type)


# -----------------------------------------------------------------------------
# Branding and Logos
# -----------------------------------------------------------------------------

COMPANY_COLORS: Dict[str, str] = {
    "ANA Seguros": "#FE1034",
    "HDI Seguros": "#006729",
    "Qualitas": "#666678",
    "Seguros Atlas": "#D0112B",
}


def generate_placeholder_logo_data_uri(company_name: str, color_hex: str) -> str:
    """Generate a simple SVG logo as data URI as a placeholder.

    Using SVG ensures crisp rendering and avoids binary files in repo.
    """
    svg = (
        f"""
        <svg xmlns='http://www.w3.org/2000/svg' width='160' height='54'>
          <rect width='100%' height='100%' fill='white'/>
          <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
                font-family='Arial, Helvetica, sans-serif' font-size='18' fill='{color_hex}'>
            {company_name}
          </text>
        </svg>
        """.strip()
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


COMPANY_LOGOS: Dict[str, str] = {
    name: generate_placeholder_logo_data_uri(name, color)
    for name, color in COMPANY_COLORS.items()
}


# -----------------------------------------------------------------------------
# Vehicle brand list (loaded from brands.json if available)
# -----------------------------------------------------------------------------


def load_brands() -> List[str]:
    brands_path = os.path.join(os.path.dirname(__file__), "brands.json")
    if os.path.exists(brands_path):
        try:
            with open(brands_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception as exc:
            logger.warning("Failed to load brands.json: %s", exc)
    # Fallback set of common brands in MX market
    return [
        "Nissan",
        "Volkswagen",
        "VW",
        "Toyota",
        "Honda",
        "Chevrolet",
        "Ford",
        "Kia",
        "Mazda",
        "Hyundai",
        "Renault",
        "SEAT",
        "Seat",
        "Dodge",
        "Jeep",
        "RAM",
        "BMW",
        "Mercedes-Benz",
        "Mercedes",
        "Audi",
        "Peugeot",
        "Suzuki",
    ]


VEHICLE_BRANDS = load_brands()


def extract_vehicle_name_from_text(text: str) -> Optional[str]:
    """Very simple heuristic to extract a vehicle description.

    Looks for a known brand followed by model words and a 4-digit year.
    """
    text_norm = re.sub(r"\s+", " ", text)
    candidates: List[str] = []

    year_match = re.search(r"(19|20)\d{2}", text_norm)
    year = year_match.group(0) if year_match else ""

    for brand in VEHICLE_BRANDS:
        pattern = rf"\b{re.escape(brand)}\b[^\n\r]{0,40}?{year}"
        m = re.search(pattern, text_norm, flags=re.IGNORECASE)
        if m:
            snippet = m.group(0)
            # Clean up snippet
            snippet = re.sub(r"\s{2,}", " ", snippet).strip()
            candidates.append(snippet)

    if candidates:
        # Choose the longest candidate as most descriptive
        return max(candidates, key=len)
    # Fallback: brand only if found
    for brand in VEHICLE_BRANDS:
        if re.search(rf"\b{re.escape(brand)}\b", text_norm, re.IGNORECASE):
            return brand
    return None


# -----------------------------------------------------------------------------
# Parsing helpers and business logic
# -----------------------------------------------------------------------------


def _to_num(value: str) -> float:
    if not value:
        return 0.0
    numeric = re.sub(r"[^0-9.,-]", "", str(value)).replace(",", "")
    try:
        return float(numeric)
    except Exception:
        return 0.0


def calculate_financials(result: Dict[str, float]) -> Dict[str, float]:
    """Calculate IVA (16%) and Prima Total."""
    try:
        prima_neta = _to_num(result.get("Prima Neta", "0"))
        recargos = _to_num(result.get("Recargos", "0"))
        derechos = _to_num(result.get("Derechos de Póliza", "0"))
        iva = (prima_neta + recargos + derechos) * 0.16
        prima_total = prima_neta + recargos + derechos + iva

        def as_currency(n: float) -> str:
            return f"${n:,.2f}"

        result["IVA"] = as_currency(iva)
        result["Prima Total"] = as_currency(prima_total)
    except Exception as exc:
        logger.error("Error calculating financials: %s", exc)
    return result


def apply_standard_defaults(result: Dict[str, float]) -> Dict[str, float]:
    defaults = {
        "Asistencia Viajes": "Amparada",
        "Atlas Cero Plus por PT de DM": "Amparada",
        "Accidente al conductor": "$100,000.00",
        "Deducible - DM": "3%",
        "Deducible - RT": "5%",
    }
    for key, default_value in defaults.items():
        if not result.get(key) or str(result.get(key)).strip().upper() in {"N/A", ""}:
            result[key] = default_value

    if result.get("company") == "Seguros Atlas":
        result["Desbielamiento por agua al motor"] = "Amparada"
    return result


def convert_to_standard_format(extracted: Dict[str, Any]) -> Dict[str, float]:
    """Map model keys to standardized Spanish labels."""
    mapping = {
        "company": "company",
        "vehicle_name": "Vehículo",
        "forma_de_pago": "Forma de Pago",
        "danos_materiales": "Daños Materiales",
        "deducible_dm": "Deducible - DM",
        "robo_total": "Robo Total",
        "deducible_rt": "Deducible - RT",
        "responsabilidad_civil": "Responsabilidad Civil",
        "gastos_medicos_ocupantes": "Gastos Medicos Ocupantes",
        "asistencia_legal": "Asistencia Legal",
        "asistencia_viajes": "Asistencia Viajes",
        "accidente_conductor": "Accidente al conductor",
        "responsabilidad_civil_catastrofica": "Responsabilidad civil catastrofica",
        "desbielamiento_agua_motor": "Desbielamiento por agua al motor",
        "prima_neta": "Prima Neta",
        "recargos": "Recargos",
        "derechos_poliza": "Derechos de Póliza",
        "iva": "IVA",
        "prima_total": "Prima Total",
    }

    result: Dict[str, float] = {}
    for src, dst in mapping.items():
        val = extracted.get(src, "") if isinstance(extracted, dict) else ""
        result[dst] = val if isinstance(val, str) else (json.dumps(val) if val is not None else "")
    return result


def identify_company_from_text_fallback(pdf_path: str) -> Optional[str]:
    """Fallback company identification using text extraction via PyMuPDF."""
    if not _PYMUPDF_AVAILABLE:
        return None
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        text_upper = text.upper()

        if re.search(r"\bSEGUROS\s+ATLAS\b", text_upper):
            return "Atlas"
        if re.search(r"\bATLAS\b", text_upper):
            return "Atlas"
        if re.search(r"\bANA\b", text_upper) and not re.search(
            r"\b(HDI|ATLAS|QUALITAS|QUÁLITAS)\b", text_upper
        ):
            return "ANA"
        if re.search(r"\bHDI(\s+SEGUROS)?\b", text_upper):
            return "HDI"
        if re.search(r"\b(QUÁLITAS|QUALITAS)\b", text_upper):
            return "Qualitas"
        return None
    except Exception:
        return None


def _safe_get_extraction(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    # agentic-doc commonly returns a list of results
    try:
        if isinstance(obj, list) and obj:
            item = obj[0]
            if isinstance(item, dict):
                return item.get("extraction") or item
            extraction = getattr(item, "extraction", None)
            if isinstance(extraction, dict):
                return extraction
        if isinstance(obj, dict):
            return obj.get("extraction") or obj
    except Exception:
        return None
    return None


def parse_pdf_with_landing_ai(pdf_content: bytes, filename: str = "document.pdf") -> Optional[Dict[str, float]]:
    """Parse PDF using Landing AI's Agentic Document Extraction."""
    if not _LANDING_AI_AVAILABLE:
        return None

    api_key = os.getenv("LANDING_AI_API_KEY")
    if not api_key:
        logger.warning("LANDING_AI_API_KEY not set; skipping Landing AI parsing")
        return None

    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_content)
            temp_path = tmp.name

        try:
            # Step 1: classification
            classification_results = landing_parse(
                temp_path, extraction_model=classification_schema, api_key=api_key
            )
            cls_extraction = _safe_get_extraction(classification_results) or {}
            document_type = str(cls_extraction.get("document_type", "Other"))

            if document_type == "Other":
                company = identify_company_from_text_fallback(temp_path)
                company_map = {
                    "HDI": "HDI Seguros",
                    "Qualitas": "Qualitas",
                    "ANA": "ANA Seguros",
                    "Atlas": "Seguros Atlas",
                }
                document_type = company_map.get(company or "", "Other")
            if document_type == "Other":
                return None

            schema = get_extraction_schema(document_type)
            if not schema:
                return None

            extraction_results = landing_parse(
                temp_path, extraction_model=schema, api_key=api_key
            )
            extraction = _safe_get_extraction(extraction_results)
            if not extraction:
                return None

            result = convert_to_standard_format(extraction)
            result = apply_standard_defaults(result)
            result = calculate_financials(result)
            # Normalize company field
            result["company"] = document_type
            # Vehicle default from text if missing
            if not result.get("Vehículo") and _PYMUPDF_AVAILABLE:
                try:
                    doc = fitz.open(temp_path)
                    text = "".join(page.get_text() for page in doc)
                    doc.close()
                    veh = extract_vehicle_name_from_text(text) or ""
                    if veh:
                        result["Vehículo"] = veh
                except Exception:
                    pass
            return result
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    except Exception as exc:
        logger.error("Error parsing PDF with Landing AI: %s", exc)
        return None


def _extract_text_from_pdf_bytes(pdf_content: bytes) -> str:
    if not _PYMUPDF_AVAILABLE:
        return ""
    try:
        import io

        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as exc:
        logger.warning("PyMuPDF failed to extract text: %s", exc)
        return ""


def parse_pdf_original_regex(pdf_content: bytes) -> Optional[Dict[str, str]]:
    """Very naive regex/text-based parsing as a last resort."""
    text = _extract_text_from_pdf_bytes(pdf_content)
    if not text:
        return None

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    haystack = "\n".join(lines)
    haystack_upper = haystack.upper()

    # Identify company roughly
    company = "Other"
    if re.search(r"\bSEGUROS\s+ATLAS\b|\bATLAS\b", haystack_upper):
        company = "Seguros Atlas"
    elif re.search(r"\bHDI(\s+SEGUROS)?\b", haystack_upper):
        company = "HDI Seguros"
    elif re.search(r"\b(QUÁLITAS|QUALITAS)\b", haystack_upper):
        company = "Qualitas"
    elif re.search(r"\bANA\b", haystack_upper):
        company = "ANA Seguros"

    def find_value(patterns: List[str]) -> str:
        for p in patterns:
            m = re.search(p, haystack, flags=re.IGNORECASE)
            if m:
                grp = next((g for g in m.groups() if g is not None), None)
                return grp.strip() if grp else m.group(0).strip()
        return "N/A"

    # Simple patterns
    value_map: Dict[str, str] = {}
    value_map["Vehículo"] = extract_vehicle_name_from_text(haystack) or ""
    value_map["Forma de Pago"] = find_value([
        r"Forma\s*de\s*Pago\s*[:\-]?\s*(.+)",
        r"Pago\s*[:\-]?\s*(Contado|Anual|Mensual|Semestral|Trimestral)"
    ])
    value_map["Daños Materiales"] = find_value([
        r"Da[nñ]os\s*Materiales\s*[:\-]?\s*([^\n]+)",
    ])
    value_map["Deducible - DM"] = find_value([
        r"Deducible\s*(?:DM|Da[nñ]os\s*Materiales)\s*[:\-]?\s*([0-9.,%]+)"
    ])
    value_map["Robo Total"] = find_value([
        r"Robo\s*Total\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Deducible - RT"] = find_value([
        r"Deducible\s*(?:RT|Robo\s*Total)\s*[:\-]?\s*([0-9.,%]+)"
    ])
    value_map["Responsabilidad Civil"] = find_value([
        r"Responsabilidad\s*Civil\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Gastos Medicos Ocupantes"] = find_value([
        r"Gastos\s*M[eé]dicos\s*Ocupantes\s*[:\-]?\s*([^\n]+)",
        r"Gastos\s*M[eé]dicos\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Asistencia Legal"] = find_value([
        r"Asistencia\s*Legal\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Asistencia Viajes"] = find_value([
        r"Asistencia\s*Viajes\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Accidente al conductor"] = find_value([
        r"Accidente\s*al\s*conductor\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Responsabilidad civil catastrofica"] = find_value([
        r"Responsabilidad\s*civil\s*catastrof[íi]ca\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Desbielamiento por agua al motor"] = find_value([
        r"Desbielamiento\s*por\s*agua\s*al\s*motor\s*[:\-]?\s*([^\n]+)"
    ])
    value_map["Prima Neta"] = find_value([
        r"Prima\s*Neta\s*[:\-]?\s*\$?\s*([0-9.,]+)"
    ])
    value_map["Recargos"] = find_value([
        r"Recargos\s*[:\-]?\s*\$?\s*([0-9.,]+)"
    ])
    value_map["Derechos de Póliza"] = find_value([
        r"Derechos\s*de\s*P[oó]liza\s*[:\-]?\s*\$?\s*([0-9.,]+)"
    ])

    # Compose final result
    result = {**value_map}
    result["company"] = company
    result = apply_standard_defaults(result)
    result = calculate_financials(result)
    return result


def parse_pdf(pdf_content: bytes) -> Optional[Dict[str, str]]:
    """Main function to parse PDF content using Landing AI with fallback."""
    try:
        landing = parse_pdf_with_landing_ai(pdf_content)
        if landing:
            return landing
    except Exception as exc:
        logger.warning("Landing AI parsing failed: %s", exc)

    logger.info("Landing AI parsing failed/unavailable, falling back to regex parser")
    return parse_pdf_original_regex(pdf_content)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_data_urlsafe_b64(data: Any) -> str:
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_data_urlsafe_b64(data_b64: str) -> Any:
    try:
        raw = base64.urlsafe_b64decode(data_b64.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to decode base64 data: %s", exc)
        raise


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        weasyprint_available=_WEASYPRINT_AVAILABLE,
        landing_ai_available=_LANDING_AI_AVAILABLE,
        supported_companies=list(COMPANY_COLORS.keys()),
    )


@app.post("/process")
def process() -> str:
    if "files" not in request.files:
        flash("No files part in request", "error")
        return redirect(url_for("index"))

    files = request.files.getlist("files")
    if not files:
        flash("No PDF files selected", "error")
        return redirect(url_for("index"))

    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for storage in files:
        filename = storage.filename or "document.pdf"
        try:
            content = storage.read()
            if not content:
                errors.append(f"{filename}: archivo vacío")
                continue

            if not filename.lower().endswith(".pdf"):
                # Attempt to detect minimal PDF signature
                if not content.startswith(b"%PDF"):
                    errors.append(f"{filename}: no es un PDF válido")
                    continue

            digest = _hash_bytes(content)
            if digest in _parse_cache:
                parsed = _parse_cache[digest]
            else:
                parsed = parse_pdf(content) or {}
                _parse_cache[digest] = parsed

            if not parsed:
                errors.append(f"{filename}: no se pudo extraer información")
                continue

            # Attach metadata
            parsed["_filename"] = filename
            color = COMPANY_COLORS.get(parsed.get("company", ""), "#444")
            parsed["_brand_color"] = color
            parsed["_logo"] = COMPANY_LOGOS.get(parsed.get("company", ""), None)

            results.append(parsed)
        except Exception as exc:
            logger.exception("Failed to process %s: %s", filename, exc)
            errors.append(f"{filename}: error procesando el archivo")

    if not results:
        flash("No se pudieron procesar los archivos seleccionados.", "error")
        return redirect(url_for("index"))

    data_b64 = encode_data_urlsafe_b64({"results": results})
    return render_template(
        "results.html",
        results=results,
        data_b64=data_b64,
        company_colors=COMPANY_COLORS,
        weasyprint_available=_WEASYPRINT_AVAILABLE,
    )


@app.get("/export/<data_b64>")
def export_pdf(data_b64: str):
    try:
        payload = decode_data_urlsafe_b64(data_b64)
    except Exception:
        abort(400, description="Datos de exportación inválidos")

    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list) or not results:
        abort(400, description="No hay datos para exportar")

    html = render_template(
        "export.html",
        results=results,
        company_colors=COMPANY_COLORS,
    )

    if _WEASYPRINT_AVAILABLE and HTML is not None:
        try:
            pdf_bytes = HTML(string=html, base_url=request.base_url).write_pdf()
            headers = {
                "Content-Type": "application/pdf",
                "Content-Disposition": "attachment; filename=comparativo.pdf",
            }
            return Response(pdf_bytes, headers=headers)
        except Exception as exc:
            logger.error("WeasyPrint failed, falling back to HTML download: %s", exc)

    # Fallback: return HTML for manual print/save as PDF
    headers = {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Disposition": "attachment; filename=comparativo.html",
    }
    return Response(html, headers=headers)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "landing_ai_available": _LANDING_AI_AVAILABLE,
        "pymupdf_available": _PYMUPDF_AVAILABLE,
        "weasyprint_available": _WEASYPRINT_AVAILABLE,
        "cache_items": len(_parse_cache),
    }


# ASGI wrapper for uvicorn
asgi_app = WsgiToAsgi(app)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


