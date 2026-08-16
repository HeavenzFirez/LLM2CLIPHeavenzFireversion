"""Visualization organ for the Sovereign pipeline.

Renders the gyroid mesh and a 4D tesseract projection into self-contained,
offline HTML files viewable in any browser. No CDN, no network fetch.
"""

from .html_renderer import (
    build_gyroid_payload,
    build_tesseract_payload,
    render_html,
    write_html,
)

__all__ = [
    "build_gyroid_payload",
    "build_tesseract_payload",
    "render_html",
    "write_html",
]
