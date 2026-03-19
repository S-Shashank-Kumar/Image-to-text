#!/usr/bin/env python3
"""
image_analyzer.py
~~~~~~~~~~~~~~~~~
Analyze images using FREE vision models via OpenRouter API.

Model    : openrouter/free  (auto-selects best available free vision model)
Provider : OpenRouter  (OpenAI-compatible API)
Cost     : 100% Free — no credit card required

Free tier limits:
    - 50 requests per day
    - 20 requests per minute

Available free vision models (--model):
    llama4-maverick  -> meta-llama/llama-4-maverick:free   (best, default)
    llama4-scout     -> meta-llama/llama-4-scout:free      (fast)
    kimi-vl          -> moonshotai/kimi-vl-a3b-thinking:free
    mistral-small    -> mistralai/mistral-small-3.1-24b-instruct:free
    qwen-vl-3b       -> qwen/qwen2.5-vl-3b-instruct:free
    auto             -> openrouter/free  (auto-picks best available)

Usage:
    python image_analyzer.py --image photo.jpg
    python image_analyzer.py --image photo.jpg --detail full
    python image_analyzer.py --image photo.jpg --detail quick --output report.md
    python image_analyzer.py --image photo.jpg --model llama4-scout

Requirements:
    pip install Pillow openai

Setup:
    1. Go to   : https://openrouter.ai/keys
    2. Sign up : free, no credit card needed
    3. Create  : click "Create Key"
    4. Set it  :
         Windows PowerShell : $env:OPENROUTER_API_KEY = "sk-or-v1-..."
         Mac / Linux        : export OPENROUTER_API_KEY="sk-or-v1-..."
    5. Run     : python image_analyzer.py --image photo.jpg
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ── Optional dependency: Pillow ───────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Required dependency: OpenAI SDK ──────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False


# =============================================================================
#  CONSTANTS
# =============================================================================

APP_NAME    = "Image Context Analyzer"
APP_VERSION = "4.0.0"

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV  = "OPENROUTER_API_KEY"

# ── Available FREE vision models on OpenRouter ────────────────────────────────
#
#  ℹ️  Free model availability changes frequently on OpenRouter.
#     "auto" always works — it intelligently picks the best available
#     free vision model for your request automatically.
#
#     To see all current free models:
#     https://openrouter.ai/models?q=free&modality=image
#
FREE_MODELS: dict[str, str] = {
    # ── Auto router — ALWAYS works, picks best free vision model ─────────────
    "auto":          "openrouter/free",                                       # ← default
    # ── Specific free vision models (may change — check OpenRouter if 404) ───
    "mistral-small": "mistralai/mistral-small-3.1-24b-instruct:free",        # 24B vision
    "gemma-3-27b":   "google/gemma-3-27b-it:free",                           # Google vision
    "gemma-3-12b":   "google/gemma-3-12b-it:free",                           # lighter Google
    "qwen-vl-3b":    "qwen/qwen2.5-vl-3b-instruct:free",                    # lightweight
    "nemotron":      "nvidia/llama-3.1-nemotron-nano-8b-v1:free",            # NVIDIA
}

DEFAULT_MODEL_ALIAS = "auto"
DEFAULT_MODEL_ID    = FREE_MODELS[DEFAULT_MODEL_ALIAS]

# ── Supported image formats ───────────────────────────────────────────────────
SUPPORTED_FORMATS: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}

OUTPUT_FORMATS = {".txt", ".md", ".json"}


# =============================================================================
#  ENUMS
# =============================================================================

class DetailLevel(str, Enum):
    """Controls how thorough the analysis prompt is."""
    QUICK    = "quick"
    STANDARD = "standard"
    FULL     = "full"


class OutputFormat(str, Enum):
    """Supported save formats."""
    TEXT     = ".txt"
    MARKDOWN = ".md"
    JSON     = ".json"


# =============================================================================
#  CUSTOM EXCEPTIONS
# =============================================================================

class ImageAnalyzerError(Exception):
    """Base exception for all image analyzer errors."""


class ImageLoadError(ImageAnalyzerError):
    """Raised when an image cannot be loaded or validated."""


class OpenRouterKeyError(ImageAnalyzerError):
    """Raised when the OpenRouter API key is missing or invalid."""


class OpenRouterError(ImageAnalyzerError):
    """Raised when an OpenRouter API call fails."""


class OutputSaveError(ImageAnalyzerError):
    """Raised when the result cannot be saved to disk."""


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class ImageMetadata:
    """Immutable metadata extracted from an image file."""
    file_name:  str
    file_path:  str
    size_kb:    float
    mime_type:  str
    width:      Optional[int] = None
    height:     Optional[int] = None
    color_mode: Optional[str] = None
    img_format: Optional[str] = None

    def __str__(self) -> str:
        dims = f"{self.width}x{self.height}" if self.width else "unknown"
        return (
            f"{self.file_name}  |  {dims}px  |  "
            f"{self.color_mode or 'N/A'}  |  {self.size_kb} KB"
        )


@dataclass
class AnalysisConfig:
    """All configuration needed to run one analysis job."""
    image_path:   str
    detail_level: DetailLevel   = DetailLevel.STANDARD
    api_key:      Optional[str] = None
    model:        str           = DEFAULT_MODEL_ID
    output_path:  Optional[str] = None
    quiet:        bool          = False


@dataclass
class AnalysisResult:
    """Holds the complete output of one analysis run."""
    image_metadata: ImageMetadata
    model:          str
    detail_level:   str
    analysis_text:  str
    elapsed_sec:    float
    prompt_tokens:  Optional[int] = None
    output_tokens:  Optional[int] = None
    timestamp:      str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "timestamp":     self.timestamp,
            "model":         self.model,
            "detail_level":  self.detail_level,
            "elapsed_sec":   self.elapsed_sec,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "image": {
                "file_name":  self.image_metadata.file_name,
                "file_path":  self.image_metadata.file_path,
                "size_kb":    self.image_metadata.size_kb,
                "mime_type":  self.image_metadata.mime_type,
                "width":      self.image_metadata.width,
                "height":     self.image_metadata.height,
                "color_mode": self.image_metadata.color_mode,
                "format":     self.image_metadata.img_format,
            },
            "analysis": self.analysis_text,
        }


# =============================================================================
#  LOGGING
# =============================================================================

def configure_logging(quiet: bool = False) -> logging.Logger:
    """
    Set up and return the module-level logger.

    Args:
        quiet: When True, suppress INFO-level output.

    Returns:
        Configured Logger instance.
    """
    log = logging.getLogger(APP_NAME)
    log.setLevel(logging.DEBUG)

    if log.handlers:
        return log  # Already configured — avoid duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING if quiet else logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    log.addHandler(handler)
    return log


logger = configure_logging()


# =============================================================================
#  PROMPTS
# =============================================================================

_PROMPTS: dict[DetailLevel, str] = {

    DetailLevel.QUICK: """\
Analyze this image and provide a concise summary covering:
  1. What the image shows (main subject or scene).
  2. Key objects or people visible.
  3. Setting or environment.
Keep your response to 3-5 sentences.""",

    DetailLevel.STANDARD: """\
Analyze this image and provide a structured report with the following sections:

SCENE OVERVIEW
Describe the main subject, scene type, and overall context.

KEY ELEMENTS
List and describe the main objects, people, animals, or elements visible.

COLORS & VISUAL STYLE
Describe dominant colors, lighting quality, and overall visual mood.

SETTING & CONTEXT
Describe the apparent location, environment, and time of day if determinable.

NOTABLE DETAILS
Note any visible text, logos, expressions, or other interesting details.

OVERALL SUMMARY
One paragraph describing the complete context and meaning of this image.""",

    DetailLevel.FULL: """\
You are an expert image analyst. Provide a comprehensive structured analysis.

1. SCENE IDENTIFICATION
   Type of image (photograph, illustration, screenshot, diagram, etc.)
   and primary subject or focus.

2. CONTENT DESCRIPTION
   All visible objects, people, animals, and elements with quantities,
   positions, and relationships.

3. PEOPLE & FACES (if present)
   Apparent age, expressions, poses, attire, and activities.

4. TEXT & SYMBOLS (if present)
   Transcribe any visible text, logos, signs, or symbols verbatim.

5. ENVIRONMENT & SETTING
   Indoor/outdoor, location type, time of day, weather, architectural details.

6. COLORS, LIGHTING & COMPOSITION
   Dominant palette, lighting source and quality, photographic composition.

7. MOOD & ATMOSPHERE
   Emotional tone and atmosphere conveyed by the image.

8. POSSIBLE USE CASE
   What this image is likely used for and what story it tells.

9. TECHNICAL QUALITY (if photograph)
   Sharpness, exposure, depth of field, apparent post-processing.

10. COMPLETE NARRATIVE
    A 2-3 paragraph description written as if for someone who cannot see
    the image.""",
}


def get_prompt(detail_level: DetailLevel) -> str:
    """
    Return the analysis prompt for the given detail level.

    Args:
        detail_level: Desired level of analysis depth.

    Returns:
        Prompt string to send to the model.
    """
    return _PROMPTS[detail_level]


# =============================================================================
#  IMAGE LOADER
# =============================================================================

def load_image(image_path: str) -> tuple[ImageMetadata, str]:
    """
    Load, validate, and base64-encode an image file.

    Args:
        image_path: Absolute or relative path to the image.

    Returns:
        Tuple of (ImageMetadata, base64-encoded image string).

    Raises:
        ImageLoadError: If the file is missing, unsupported, or corrupted.
    """
    path = Path(image_path).resolve()

    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")

    if not path.is_file():
        raise ImageLoadError(f"Path is not a file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ImageLoadError(
            f"Unsupported format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    mime_type = SUPPORTED_FORMATS[ext]
    raw_bytes = path.read_bytes()

    if len(raw_bytes) == 0:
        raise ImageLoadError(f"Image file is empty: {path}")

    b64_data = base64.b64encode(raw_bytes).decode("utf-8")
    size_kb  = round(len(raw_bytes) / 1024, 1)

    width = height = color_mode = img_format = None

    if PIL_AVAILABLE:
        try:
            with PILImage.open(path) as img:
                img.verify()
            with PILImage.open(path) as img:
                width      = img.size[0]
                height     = img.size[1]
                color_mode = img.mode
                img_format = img.format or ext.lstrip(".").upper()
        except Exception as exc:
            raise ImageLoadError(
                f"Image is corrupted or unreadable: {exc}"
            ) from exc
    else:
        logger.warning("Pillow not installed — image metadata unavailable.")

    return ImageMetadata(
        file_name  = path.name,
        file_path  = str(path),
        size_kb    = size_kb,
        mime_type  = mime_type,
        width      = width,
        height     = height,
        color_mode = color_mode,
        img_format = img_format,
    ), b64_data


# =============================================================================
#  OPENROUTER CLIENT
# =============================================================================

class OpenRouterClient:
    """
    Client for the OpenRouter API using the OpenAI-compatible interface.

    OpenRouter normalizes requests and responses across all providers,
    so we use the standard OpenAI SDK — just with a different base_url.

    Free tier:
        - 50 requests per day
        - 20 requests per minute
        - No credit card required

    Get a free key at: https://openrouter.ai/keys

    Attributes:
        api_key: OpenRouter API key (sk-or-v1-...).
        model:   Full model ID on OpenRouter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model:   str           = DEFAULT_MODEL_ID,
    ) -> None:
        self.model   = model
        self.api_key = api_key or os.environ.get(OPENROUTER_KEY_ENV)

        if not self.api_key:
            raise OpenRouterKeyError(
                "OpenRouter API key not found.\n\n"
                "  HOW TO GET A FREE KEY (no credit card needed):\n"
                "    1. Go to   : https://openrouter.ai/keys\n"
                "    2. Sign up : takes 30 seconds\n"
                "    3. Click   : 'Create Key'\n"
                "    4. Copy    : key starts with sk-or-v1-...\n\n"
                "  HOW TO SET IT:\n"
                "    Windows PowerShell:\n"
                '      $env:OPENROUTER_API_KEY = "sk-or-v1-..."\n\n'
                "    Mac / Linux:\n"
                '      export OPENROUTER_API_KEY="sk-or-v1-..."\n\n'
                "  OR pass it directly:\n"
                "    python image_analyzer.py --image photo.jpg --api-key sk-or-v1-..."
            )

        if not OPENAI_SDK_AVAILABLE:
            raise OpenRouterError(
                "openai package not installed.\n"
                "  Run: pip install openai"
            )

        # OpenRouter is OpenAI-compatible — only base_url changes
        self._client = OpenAI(
            api_key  = self.api_key,
            base_url = OPENROUTER_BASE_URL,
        )

    def analyze(
        self,
        b64_image: str,
        mime_type: str,
        prompt:    str,
    ) -> tuple[str, Optional[int], Optional[int]]:
        """
        Send an image to the free vision model and return the analysis.

        Args:
            b64_image: Base64-encoded image data.
            mime_type: MIME type of the image (e.g. 'image/jpeg').
            prompt:    Instruction prompt for the model.

        Returns:
            Tuple of (analysis_text, prompt_tokens, output_tokens).

        Raises:
            OpenRouterKeyError: If the API key is invalid or expired.
            OpenRouterError:    If the API call fails.
        """
        try:
            response = self._client.chat.completions.create(
                model      = self.model,
                max_tokens = 2048,
                messages   = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_image}"
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            text   = response.choices[0].message.content.strip()
            p_tok  = response.usage.prompt_tokens     if response.usage else None
            o_tok  = response.usage.completion_tokens if response.usage else None

            return text, p_tok, o_tok

        except Exception as exc:
            msg = str(exc).lower()

            if "401" in msg or "unauthorized" in msg or "invalid" in msg:
                raise OpenRouterKeyError(
                    "Invalid or expired OpenRouter API key.\n"
                    "  Check your key at: https://openrouter.ai/keys"
                ) from exc

            if "429" in msg or "rate" in msg:
                raise OpenRouterError(
                    "Rate limit hit.\n"
                    "  Free tier: 20 req/min, 50 req/day.\n"
                    "  Wait a minute and try again."
                ) from exc

            if "402" in msg or "credits" in msg:
                raise OpenRouterError(
                    "Credit error — make sure you are using a :free model.\n"
                    f"  Current model: {self.model}"
                ) from exc

            if "connection" in msg or "timeout" in msg or "network" in msg:
                raise OpenRouterError(
                    "Network error reaching OpenRouter.\n"
                    "  Check your internet connection and try again."
                ) from exc

            raise OpenRouterError(f"OpenRouter API error: {exc}") from exc


# =============================================================================
#  OUTPUT WRITER
# =============================================================================

class OutputWriter:
    """Writes an AnalysisResult to disk in the requested format."""

    @staticmethod
    def save(result: AnalysisResult, output_path: str) -> None:
        """
        Persist the analysis result to a file.

        Args:
            result:      The completed analysis result.
            output_path: Destination file path (.txt, .md, or .json).

        Raises:
            OutputSaveError: If the extension is unsupported or write fails.
        """
        path = Path(output_path)
        ext  = path.suffix.lower()

        if ext not in OUTPUT_FORMATS:
            raise OutputSaveError(
                f"Unsupported output format '{ext}'. "
                f"Use one of: {sorted(OUTPUT_FORMATS)}"
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            {
                ".json": OutputWriter._write_json,
                ".md":   OutputWriter._write_markdown,
                ".txt":  OutputWriter._write_text,
            }[ext](result, path)
        except OSError as exc:
            raise OutputSaveError(
                f"Could not write to {path}: {exc}"
            ) from exc

    @staticmethod
    def _write_json(result: AnalysisResult, path: Path) -> None:
        path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _write_markdown(result: AnalysisResult, path: Path) -> None:
        meta  = result.image_metadata
        lines = [
            "# Image Analysis Report\n",
            "| Field | Value |",
            "|---|---|",
            f"| **Image**         | `{meta.file_name}` |",
            f"| **Model**         | {result.model} |",
            f"| **Detail Level**  | {result.detail_level} |",
            f"| **Analyzed At**   | {result.timestamp} |",
            f"| **Elapsed**       | {result.elapsed_sec}s |",
            f"| **Prompt Tokens** | {result.prompt_tokens or 'N/A'} |",
            f"| **Output Tokens** | {result.output_tokens or 'N/A'} |",
            "\n## Image Metadata\n",
            "| Property | Value |",
            "|---|---|",
            f"| Dimensions | {meta.width}x{meta.height} px |",
            f"| Color Mode | {meta.color_mode} |",
            f"| Format     | {meta.img_format} |",
            f"| File Size  | {meta.size_kb} KB |",
            "\n## Analysis\n",
            result.analysis_text,
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _write_text(result: AnalysisResult, path: Path) -> None:
        meta  = result.image_metadata
        lines = [
            f"{'=' * 60}",
            f"  {APP_NAME.upper()}  —  v{APP_VERSION}",
            f"{'=' * 60}",
            f"Image          : {meta.file_name}",
            f"Dimensions     : {meta.width}x{meta.height} px",
            f"File Size      : {meta.size_kb} KB",
            f"Model          : {result.model}",
            f"Detail Level   : {result.detail_level}",
            f"Timestamp      : {result.timestamp}",
            f"Elapsed        : {result.elapsed_sec}s",
            f"Prompt Tokens  : {result.prompt_tokens or 'N/A'}",
            f"Output Tokens  : {result.output_tokens or 'N/A'}",
            f"{'─' * 60}",
            "",
            result.analysis_text,
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
#  CONSOLE RENDERER
# =============================================================================

class ConsoleRenderer:
    """
    Renders structured output to the terminal using ANSI colour codes.
    All output is suppressed when quiet=True except errors.
    """

    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"
    _CYAN   = "\033[96m"
    _GREEN  = "\033[92m"
    _YELLOW = "\033[93m"
    _RED    = "\033[91m"
    _BLUE   = "\033[94m"

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def header(self) -> None:
        if self.quiet:
            return
        print(
            f"\n{self._CYAN}{self._BOLD}"
            f"╔{'═' * 56}╗\n"
            f"║{'':^8}{APP_NAME:^40}{'':^8}║\n"
            f"║{'':^8}{'v' + APP_VERSION + '  ·  OpenRouter  ·  FREE':^40}{'':^8}║\n"
            f"╚{'═' * 56}╝"
            f"{self._RESET}\n"
        )

    def section(self, title: str, icon: str = "►") -> None:
        if self.quiet:
            return
        print(f"\n{self._YELLOW}{self._BOLD}{icon}  {title}{self._RESET}")
        print(f"{self._DIM}{'─' * 56}{self._RESET}")

    def key_value(self, key: str, value: str) -> None:
        if self.quiet:
            return
        print(f"  {self._CYAN}{key:<24}{self._RESET}{value}")

    def success(self, message: str) -> None:
        if self.quiet:
            return
        print(f"  {self._GREEN}✔  {message}{self._RESET}")

    def info(self, message: str) -> None:
        if self.quiet:
            return
        print(f"  {self._BLUE}ℹ  {message}{self._RESET}")

    def error(self, message: str) -> None:
        # Always printed regardless of quiet mode
        print(f"  {self._RED}✖  {message}{self._RESET}", file=sys.stderr)

    def analysis(self, text: str) -> None:
        if self.quiet:
            return
        print()
        for line in text.strip().splitlines():
            if line.strip():
                print(
                    f"{self._GREEN}"
                    + textwrap.fill(
                        line, width=70,
                        initial_indent="  ",
                        subsequent_indent="  ",
                    )
                    + self._RESET
                )
            else:
                print()
        print()


# =============================================================================
#  MAIN ANALYZER
# =============================================================================

class ImageAnalyzer:
    """
    Orchestrates image loading, OpenRouter inference,
    result formatting, and output persistence.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config   = config
        self._renderer = ConsoleRenderer(quiet=config.quiet)
        self._client   = OpenRouterClient(
            api_key = config.api_key,
            model   = config.model,
        )

    def run(self) -> AnalysisResult:
        """
        Execute the full analysis pipeline.

        Returns:
            Completed AnalysisResult.

        Raises:
            ImageLoadError:     If the image cannot be loaded.
            OpenRouterKeyError: If the API key is missing or invalid.
            OpenRouterError:    If the API call fails.
            OutputSaveError:    If the result cannot be saved.
        """
        renderer = self._renderer
        config   = self._config

        renderer.header()

        # ── Step 1: Load image ────────────────────────────────────────────────
        renderer.section("LOADING IMAGE", "📷")
        metadata, b64_image = load_image(config.image_path)
        renderer.success(f"Loaded: {metadata}")

        # ── Step 2: Show configuration ────────────────────────────────────────
        renderer.section("CONFIGURATION", "🤖")
        renderer.key_value("Model:",        config.model)
        renderer.key_value("Cost:",         "FREE  ✅  (OpenRouter free tier)")
        renderer.key_value("Detail Level:", config.detail_level.value.upper())

        # ── Step 3: Build prompt ──────────────────────────────────────────────
        prompt = get_prompt(config.detail_level)

        # ── Step 4: Run inference ─────────────────────────────────────────────
        renderer.section("ANALYZING IMAGE", "🔍")
        renderer.info("Sending to OpenRouter — auto-routing to best free vision model …")

        start                         = time.perf_counter()
        analysis_text, p_tok, o_tok   = self._client.analyze(
            b64_image, metadata.mime_type, prompt
        )
        elapsed = round(time.perf_counter() - start, 2)

        # ── Step 5: Display results ───────────────────────────────────────────
        renderer.section("ANALYSIS RESULT", "✨")
        renderer.key_value("Time Taken:",     f"{elapsed}s")
        renderer.key_value("Prompt Tokens:",  str(p_tok or "N/A"))
        renderer.key_value("Output Tokens:",  str(o_tok or "N/A"))
        renderer.analysis(analysis_text)

        # ── Step 6: Build result object ───────────────────────────────────────
        result = AnalysisResult(
            image_metadata = metadata,
            model          = config.model,
            detail_level   = config.detail_level.value,
            analysis_text  = analysis_text,
            elapsed_sec    = elapsed,
            prompt_tokens  = p_tok,
            output_tokens  = o_tok,
        )

        # ── Step 7: Save output if requested ─────────────────────────────────
        if config.output_path:
            renderer.section("SAVING OUTPUT", "💾")
            OutputWriter.save(result, config.output_path)
            renderer.success(f"Saved → {config.output_path}")

        renderer.section("DONE", "🎉")
        renderer.success(f"Analysis complete in {elapsed}s")
        return result


# =============================================================================
#  CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""

    model_list = "\n".join(
        f"  {alias:<14} → {model_id}"
        for alias, model_id in FREE_MODELS.items()
    )

    parser = argparse.ArgumentParser(
        prog            = "image_analyzer.py",
        description     = f"🖼️  {APP_NAME} — 100% Free via OpenRouter + Qwen2.5 VL",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = f"""
SETUP
  1. Get free key : https://openrouter.ai/keys  (no credit card)
  2. Install      : pip install Pillow openai
  3. Set key      : $env:OPENROUTER_API_KEY = "sk-or-v1-..."
  4. Run          : python image_analyzer.py --image photo.jpg

EXAMPLES
  Standard analysis (default):
    python image_analyzer.py --image photo.jpg

  Deep analysis saved to Markdown:
    python image_analyzer.py --image photo.jpg --detail full --output report.md

  Quick summary as JSON:
    python image_analyzer.py --image photo.jpg --detail quick --output result.json

  Use a different free model:
    python image_analyzer.py --image photo.jpg --model qwen-32b
    python image_analyzer.py --image photo.jpg --model llama-vision

  Pass API key directly:
    python image_analyzer.py --image photo.jpg --api-key sk-or-v1-...

DETAIL LEVELS
  quick    → 3-5 sentence summary
  standard → 6-section structured report     [DEFAULT]
  full     → 10-section deep analysis + narrative

FREE MODELS (--model)
{model_list}

OUTPUT FORMATS
  .md    → Markdown report with metadata table
  .json  → Machine-readable JSON
  .txt   → Plain text report

FREE TIER LIMITS
  50 requests per day  |  20 requests per minute  |  No credit card needed
        """,
    )

    parser.add_argument(
        "--image", "-i",
        required = True,
        metavar  = "PATH",
        help     = "Path to the image file to analyze",
    )
    parser.add_argument(
        "--detail", "-d",
        default  = DetailLevel.STANDARD.value,
        choices  = [d.value for d in DetailLevel],
        metavar  = "{quick,standard,full}",
        help     = "Analysis depth (default: standard)",
    )
    parser.add_argument(
        "--model", "-m",
        default  = DEFAULT_MODEL_ALIAS,
        metavar  = "NAME",
        help     = f"Free model alias or full model ID (default: {DEFAULT_MODEL_ALIAS})",
    )
    parser.add_argument(
        "--api-key", "-k",
        default  = None,
        metavar  = "KEY",
        help     = f"OpenRouter API key (or set {OPENROUTER_KEY_ENV} env var)",
    )
    parser.add_argument(
        "--output", "-o",
        default  = None,
        metavar  = "FILE",
        help     = "Save result to file (.txt, .md, or .json)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action   = "store_true",
        help     = "Suppress all decorative console output",
    )
    parser.add_argument(
        "--version", "-v",
        action   = "version",
        version  = f"%(prog)s {APP_VERSION}",
    )
    return parser


def _resolve_model(model_arg: str) -> str:
    """
    Resolve a model alias or full model ID.

    Args:
        model_arg: Short alias (e.g. 'qwen-72b') or full ID.

    Returns:
        Full OpenRouter model ID string.
    """
    return FREE_MODELS.get(model_arg, model_arg)


def main() -> None:
    """Entry point: parse arguments, build config, run analyzer."""
    parser = _build_parser()
    args   = parser.parse_args()

    config = AnalysisConfig(
        image_path   = args.image,
        detail_level = DetailLevel(args.detail),
        api_key      = args.api_key,
        model        = _resolve_model(args.model),
        output_path  = args.output,
        quiet        = args.quiet,
    )

    try:
        ImageAnalyzer(config).run()
    except ImageAnalyzerError as exc:
        ConsoleRenderer().error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()