"""Interactive HTML report generator for PDF extraction evaluation.

Renders PDF pages with bbox overlays for blocks and chunks, and generates
a self-contained HTML file with annotation tools for ground truth creation.
"""

import base64
import html
import io
import json
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# Block type colors (RGBA with semi-transparent alpha)
BLOCK_TYPE_COLORS = {
    "heading": (66, 133, 244, 80),
    "paragraph": (158, 158, 158, 80),
    "list_item": (156, 39, 176, 80),
    "table": (76, 175, 80, 80),
}
DEFAULT_BLOCK_COLOR = (255, 152, 0, 80)

# Cycling palette for chunk overlays
CHUNK_COLORS = [
    (66, 133, 244, 80),
    (234, 67, 53, 80),
    (251, 188, 4, 80),
    (52, 168, 83, 80),
    (156, 39, 176, 80),
    (255, 109, 0, 80),
]


def render_page_image(page: fitz.Page, dpi: int = 150) -> bytes:
    """Render a PDF page to PNG bytes.

    Args:
        page: A PyMuPDF page object.
        dpi: Rendering resolution in dots per inch.

    Returns:
        PNG image bytes.
    """
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")


def compute_bbox_to_image_coords(
    bbox: list[float],
    dpi: int,
    page_width: float,
    page_height: float,
    is_normalized: bool,
) -> tuple[float, float, float, float]:
    """Convert a bounding box to pixel coordinates in the rendered image.

    Args:
        bbox: [x0, y0, x1, y1] coordinates.
        dpi: Rendering DPI used for the page image.
        page_width: Page width in PDF points.
        page_height: Page height in PDF points.
        is_normalized: True if bbox values are 0-1 normalized (Reducto format).

    Returns:
        (x0, y0, x1, y1) in pixel coordinates.
    """
    x0, y0, x1, y1 = bbox
    scale = dpi / 72

    if is_normalized:
        # Reducto: normalized 0-1, multiply by page dimensions first
        x0 = x0 * page_width * scale
        y0 = y0 * page_height * scale
        x1 = x1 * page_width * scale
        y1 = y1 * page_height * scale
    else:
        # PyMuPDF: already in PDF points
        x0 = x0 * scale
        y0 = y0 * scale
        x1 = x1 * scale
        y1 = y1 * scale

    return (x0, y0, x1, y1)


def draw_block_overlays(page: fitz.Page, parsed_page, dpi: int, is_normalized: bool) -> bytes:
    """Render a page image with colored block bounding box overlays.

    Args:
        page: A PyMuPDF page object.
        parsed_page: A ParsedPage with .blocks list.
        dpi: Rendering DPI.
        is_normalized: True if bboxes are normalized (Reducto).

    Returns:
        PNG image bytes with overlays drawn.
    """
    png_bytes = render_page_image(page, dpi)
    base_image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    page_width = page.rect.width
    page_height = page.rect.height

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for block in parsed_page.blocks:
        if block.bbox is None:
            continue

        coords = compute_bbox_to_image_coords(
            block.bbox, dpi, page_width, page_height, is_normalized
        )
        color = BLOCK_TYPE_COLORS.get(block.block_type, DEFAULT_BLOCK_COLOR)
        draw.rectangle(coords, fill=color, outline=color[:3] + (180,))

        label = f"{block.block_index}"
        draw.text((coords[0] + 2, coords[1] + 2), label, fill=(0, 0, 0, 220), font=font)

    result = Image.alpha_composite(base_image, overlay)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def draw_chunk_overlays(
    page: fitz.Page,
    chunks: list,
    page_number: int,
    dpi: int,
    is_normalized: bool,
) -> bytes:
    """Render a page image with colored chunk bounding box overlays.

    Args:
        page: A PyMuPDF page object.
        chunks: List of ChunkData objects.
        page_number: 1-indexed page number to filter chunks.
        dpi: Rendering DPI.
        is_normalized: True if bboxes are normalized (Reducto).

    Returns:
        PNG image bytes with overlays drawn.
    """
    png_bytes = render_page_image(page, dpi)
    base_image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    page_width = page.rect.width
    page_height = page.rect.height

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    page_chunks = [c for c in chunks if c.page_number == page_number]

    for i, chunk in enumerate(page_chunks):
        color = CHUNK_COLORS[i % len(CHUNK_COLORS)]
        label = f"C{chunk.position}"
        labeled = False

        # Draw individual block bboxes if available, otherwise single bbox
        bboxes = chunk.block_bboxes if chunk.block_bboxes else ([chunk.bbox] if chunk.bbox else [])
        for bbox in bboxes:
            coords = compute_bbox_to_image_coords(
                bbox, dpi, page_width, page_height, is_normalized
            )
            draw.rectangle(coords, fill=color, outline=color[:3] + (180,))
            if not labeled:
                draw.text((coords[0] + 2, coords[1] + 2), label, fill=(0, 0, 0, 220), font=font)
                labeled = True

    result = Image.alpha_composite(base_image, overlay)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def generate_html_report(
    pdf_path: str | Path,
    results: dict,
    file_hash: str,
    dpi: int = 150,
) -> str:
    """Generate a self-contained interactive HTML report.

    Args:
        pdf_path: Path to the source PDF file.
        results: Dict mapping parser name to {"doc": ParsedDocument, "chunks": list[ChunkData]}.
        file_hash: SHA-256 hash of the PDF file.
        dpi: Rendering DPI for page images.

    Returns:
        Complete HTML string for the report.
    """
    pdf_path = Path(pdf_path)
    file_name = pdf_path.name
    escaped_file_name = html.escape(file_name)
    parser_names = list(results.keys())

    doc = fitz.open(pdf_path)
    try:
        # Determine if each parser uses normalized bboxes
        normalized_map = {
            "pymupdf": False,
        }

        # Pre-render all images and build report data
        report_data = {
            "file_name": file_name,
            "file_hash": file_hash,
            "parsers": {},
        }

        # Maps: parser_name -> page_number -> {"block_img_b64": ..., "chunk_img_b64": ...}
        images = {}

        for parser_name, parser_result in results.items():
            parsed_doc = parser_result["doc"]
            chunks = parser_result["chunks"]
            is_normalized = normalized_map.get(parser_name, False)

            images[parser_name] = {}
            report_data["parsers"][parser_name] = {"pages": []}

            for parsed_page in parsed_doc.pages:
                page_num = parsed_page.page_number
                page_idx = page_num - 1

                if page_idx < 0 or page_idx >= doc.page_count:
                    continue

                fitz_page = doc[page_idx]

                # Render block overlay
                block_img_bytes = draw_block_overlays(fitz_page, parsed_page, dpi, is_normalized)
                block_img_b64 = base64.b64encode(block_img_bytes).decode("ascii")

                # Render chunk overlay
                chunk_img_bytes = draw_chunk_overlays(
                    fitz_page, chunks, page_num, dpi, is_normalized
                )
                chunk_img_b64 = base64.b64encode(chunk_img_bytes).decode("ascii")

                images.setdefault(parser_name, {})[page_num] = {
                    "block_img_b64": block_img_b64,
                    "chunk_img_b64": chunk_img_b64,
                }

                # Build page data for JS
                page_blocks = []
                for block in parsed_page.blocks:
                    page_blocks.append({
                        "block_index": block.block_index,
                        "block_type": block.block_type,
                        "text": block.text,
                        "font_size": block.font_size,
                        "is_bold": block.is_bold,
                        "bbox": block.bbox,
                    })

                page_chunks = []
                for chunk in chunks:
                    if chunk.page_number == page_num:
                        page_chunks.append({
                            "content": chunk.content,
                            "chunk_type": chunk.chunk_type,
                            "page_number": chunk.page_number,
                            "position": chunk.position,
                            "bbox": chunk.bbox,
                        })

                report_data["parsers"][parser_name]["pages"].append({
                    "page_number": page_num,
                    "blocks": page_blocks,
                    "chunks": page_chunks,
                })
    finally:
        doc.close()

    # Escape </ sequences to prevent script tag breakout from PDF text content
    report_data_json = json.dumps(report_data).replace("</", "<\\/")


    # Build page sections HTML
    page_sections_html = ""
    for parser_name in parser_names:
        parser_images = images.get(parser_name, {})
        parser_pages = report_data["parsers"][parser_name]["pages"]

        for page_info in parser_pages:
            page_num = page_info["page_number"]
            img_data = parser_images.get(page_num, {})
            block_b64 = img_data.get("block_img_b64", "")
            chunk_b64 = img_data.get("chunk_img_b64", "")

            blocks_html = ""
            for block in page_info["blocks"]:
                escaped_text = (
                    block["text"]
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )

                blocks_html += f"""
                <div class="card" data-block-index="{block['block_index']}" data-block-type="{block['block_type']}">
                    <div class="card-header">
                        <span class="badge badge-{block['block_type']}">{block['block_type']}</span>
                        <span class="block-meta">Block {block['block_index']} | font: {block['font_size']:.1f} | bold: {str(block['is_bold']).lower()}</span>
                    </div>
                    <div class="card-text">{escaped_text}</div>
                    <div class="verdict-row">
                        <label><input type="radio" name="verdict-{parser_name}-{page_num}-{block['block_index']}" value="correct" onchange="onVerdictChange(this)"> Correct</label>
                        <label><input type="radio" name="verdict-{parser_name}-{page_num}-{block['block_index']}" value="partial" onchange="onVerdictChange(this)"> Partial</label>
                        <label><input type="radio" name="verdict-{parser_name}-{page_num}-{block['block_index']}" value="wrong" onchange="onVerdictChange(this)"> Wrong</label>
                    </div>
                    <textarea class="correction-input" placeholder="Corrected text (if partial or wrong)..." style="display:none;"></textarea>
                </div>"""

            chunks_html = ""
            for chunk in page_info["chunks"]:
                escaped_content = (
                    chunk["content"]
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )

                chunks_html += f"""
                <div class="card chunk-card">
                    <div class="card-header">
                        <span class="badge badge-{chunk['chunk_type']}">{chunk['chunk_type']}</span>
                        <span class="block-meta">Chunk pos {chunk['position']}</span>
                    </div>
                    <div class="card-text">{escaped_content}</div>
                </div>"""

            is_first_page = page_num == parser_pages[0]["page_number"]
            display = "block" if is_first_page else "none"

            page_sections_html += f"""
            <div class="page-section" data-parser="{parser_name}" data-page="{page_num}" style="display:{display};">
                <div class="view-toggle">
                    <button class="view-btn active" onclick="switchView('block', '{parser_name}', {page_num})">Block View</button>
                    <button class="view-btn" onclick="switchView('chunk', '{parser_name}', {page_num})">Chunk View</button>
                </div>
                <div class="two-panel">
                    <div class="left-panel">
                        <img class="page-img block-img" src="data:image/png;base64,{block_b64}" alt="Block overlay page {page_num}">
                        <img class="page-img chunk-img" src="data:image/png;base64,{chunk_b64}" alt="Chunk overlay page {page_num}" style="display:none;">
                    </div>
                    <div class="right-panel">
                        <div class="block-view">{blocks_html}</div>
                        <div class="chunk-view" style="display:none;">{chunks_html}</div>
                        <div class="missing-blocks-section block-view">
                            <h3>Add Missing Block</h3>
                            <div class="missing-block-form">
                                <select class="missing-type-select">
                                    <option value="heading">heading</option>
                                    <option value="paragraph" selected>paragraph</option>
                                    <option value="list_item">list_item</option>
                                    <option value="table">table</option>
                                </select>
                                <textarea class="missing-text-input" placeholder="Expected text..."></textarea>
                                <button onclick="addMissingBlock({page_num}, '{parser_name}')">Add Missing Block</button>
                            </div>
                            <div class="missing-blocks-list"></div>
                        </div>
                        <div class="page-notes-section">
                            <label>Page Notes:</label>
                            <textarea class="page-notes" placeholder="Notes for this page..."></textarea>
                        </div>
                    </div>
                </div>
            </div>"""

    # Build parser tabs
    parser_tabs_html = ""
    for i, name in enumerate(parser_names):
        active_class = " active" if i == 0 else ""
        parser_tabs_html += (
            f'<button class="parser-tab{active_class}" '
            f"onclick=\"switchParser('{name}')\">{name}</button>"
        )

    # Build page options
    total_pages = 0
    first_parser = parser_names[0]
    first_pages = report_data["parsers"][first_parser]["pages"]
    total_pages = len(first_pages)

    page_options = ""
    for page_info in first_pages:
        page_options += f'<option value="{page_info["page_number"]}">Page {page_info["page_number"]}</option>'

    html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PDF Extraction Report — {escaped_file_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #1a1a2e; color: #fff; padding: 16px 24px; }}
.header h1 {{ font-size: 18px; margin-bottom: 4px; }}
.header .meta {{ font-size: 12px; color: #aaa; }}
.parser-tabs {{ display: flex; gap: 4px; padding: 8px 24px; background: #16213e; }}
.parser-tab {{ padding: 8px 16px; border: none; background: transparent; color: #aaa; cursor: pointer; border-radius: 4px 4px 0 0; font-size: 14px; }}
.parser-tab.active {{ background: #f5f5f5; color: #333; }}
.page-nav {{ position: sticky; top: 0; z-index: 100; display: flex; align-items: center; gap: 12px; padding: 10px 24px; background: #fff; border-bottom: 1px solid #ddd; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.page-nav button {{ padding: 6px 14px; border: 1px solid #ccc; background: #fff; cursor: pointer; border-radius: 4px; }}
.page-nav button:hover {{ background: #eee; }}
.page-nav select {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; }}
.view-toggle {{ display: flex; gap: 4px; padding: 10px 0; }}
.view-btn {{ padding: 6px 14px; border: 1px solid #ccc; background: #fff; cursor: pointer; border-radius: 4px; font-size: 13px; }}
.view-btn.active {{ background: #1a73e8; color: #fff; border-color: #1a73e8; }}
.two-panel {{ display: flex; gap: 16px; padding: 0 24px 24px; }}
.left-panel {{ flex: 0 0 60%; max-width: 60%; }}
.right-panel {{ flex: 0 0 38%; max-width: 38%; overflow-y: auto; max-height: 85vh; }}
.page-img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
.card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; margin-bottom: 10px; }}
.card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; color: #fff; }}
.badge-heading {{ background: #4285f4; }}
.badge-paragraph {{ background: #9e9e9e; }}
.badge-list_item {{ background: #9c27b0; }}
.badge-table {{ background: #4caf50; }}
.block-meta {{ font-size: 11px; color: #888; }}
.card-text {{ font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; padding: 6px; background: #fafafa; border-radius: 4px; margin-bottom: 8px; }}
.verdict-row {{ display: flex; gap: 12px; margin-bottom: 6px; }}
.verdict-row label {{ font-size: 13px; cursor: pointer; display: flex; align-items: center; gap: 4px; }}
.correction-input {{ width: 100%; min-height: 60px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; resize: vertical; }}
.chunk-card {{ border-left: 3px solid #1a73e8; }}
.missing-blocks-section {{ margin-top: 16px; padding-top: 16px; border-top: 1px solid #eee; }}
.missing-blocks-section h3 {{ font-size: 14px; margin-bottom: 8px; }}
.missing-block-form {{ display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }}
.missing-block-form select, .missing-block-form textarea {{ padding: 6px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; }}
.missing-block-form textarea {{ min-height: 50px; resize: vertical; }}
.missing-block-form button {{ padding: 6px 14px; background: #1a73e8; color: #fff; border: none; border-radius: 4px; cursor: pointer; align-self: flex-start; }}
.missing-block-item {{ background: #fff3e0; border: 1px solid #ffe0b2; border-radius: 6px; padding: 10px; margin-bottom: 6px; }}
.page-notes-section {{ margin-top: 12px; }}
.page-notes-section label {{ font-size: 13px; font-weight: 600; display: block; margin-bottom: 4px; }}
.page-notes {{ width: 100%; min-height: 50px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; resize: vertical; }}
.footer {{ position: sticky; bottom: 0; background: #fff; border-top: 1px solid #ddd; padding: 12px 24px; display: flex; align-items: center; gap: 16px; box-shadow: 0 -1px 3px rgba(0,0,0,0.1); z-index: 100; }}
.footer input {{ padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; }}
.footer textarea {{ flex: 1; padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; min-height: 36px; resize: vertical; }}
.footer button {{ padding: 8px 20px; background: #34a853; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: 600; }}
.footer button:hover {{ background: #2d8f47; }}
.page-section {{ padding-top: 10px; }}
</style>
</head>
<body>

<div class="header">
    <h1>PDF Extraction Report</h1>
    <div class="meta">{escaped_file_name} | hash: {file_hash[:16]}... | parser(s): {', '.join(parser_names)}</div>
</div>

<div class="parser-tabs" id="parserTabs">{parser_tabs_html}</div>

<div class="page-nav">
    <button onclick="showPage('prev')">Prev</button>
    <select id="pageSelect" onchange="showPage(parseInt(this.value))">{page_options}</select>
    <button onclick="showPage('next')">Next</button>
    <span id="pageInfo">Page 1 of {total_pages}</span>
</div>

<div id="pageContainer">{page_sections_html}</div>

<div class="footer">
    <label>Annotator:</label>
    <input type="text" id="annotatorName" placeholder="Your name">
    <textarea id="docNotes" placeholder="Document-level notes..."></textarea>
    <button onclick="downloadGroundTruth()">Download Ground Truth JSON</button>
</div>

<script>
var REPORT_DATA = {report_data_json};

var currentParser = '{first_parser}';
var currentPage = {first_pages[0]["page_number"] if first_pages else 1};
var totalPages = {total_pages};

function getParserPages(parserName) {{
    var p = REPORT_DATA.parsers[parserName];
    return p ? p.pages.map(function(pg) {{ return pg.page_number; }}) : [];
}}

function showPage(target) {{
    var pages = getParserPages(currentParser);
    if (!pages.length) return;
    var idx = pages.indexOf(currentPage);

    if (target === 'prev') {{
        idx = Math.max(0, idx - 1);
    }} else if (target === 'next') {{
        idx = Math.min(pages.length - 1, idx + 1);
    }} else {{
        idx = pages.indexOf(target);
        if (idx === -1) idx = 0;
    }}

    currentPage = pages[idx];

    // Hide all page sections
    var sections = document.querySelectorAll('.page-section');
    for (var i = 0; i < sections.length; i++) {{
        sections[i].style.display = 'none';
    }}

    // Show current parser + page
    var sel = '.page-section[data-parser="' + currentParser + '"][data-page="' + currentPage + '"]';
    var active = document.querySelector(sel);
    if (active) active.style.display = 'block';

    document.getElementById('pageSelect').value = currentPage;
    document.getElementById('pageInfo').textContent = 'Page ' + currentPage + ' of ' + pages.length;
}}

function switchParser(parserName) {{
    currentParser = parserName;
    var tabs = document.querySelectorAll('.parser-tab');
    for (var i = 0; i < tabs.length; i++) {{
        tabs[i].classList.remove('active');
        if (tabs[i].textContent === parserName) tabs[i].classList.add('active');
    }}

    // Update page select options
    var pages = getParserPages(parserName);
    var select = document.getElementById('pageSelect');
    select.innerHTML = '';
    for (var j = 0; j < pages.length; j++) {{
        var opt = document.createElement('option');
        opt.value = pages[j];
        opt.textContent = 'Page ' + pages[j];
        select.appendChild(opt);
    }}
    totalPages = pages.length;

    if (pages.indexOf(currentPage) === -1) {{
        currentPage = pages[0] || 1;
    }}
    showPage(currentPage);
}}

function switchView(view, parserName, pageNum) {{
    var section = document.querySelector('.page-section[data-parser="' + parserName + '"][data-page="' + pageNum + '"]');
    if (!section) return;

    var btns = section.querySelectorAll('.view-btn');
    for (var i = 0; i < btns.length; i++) {{
        btns[i].classList.remove('active');
        if ((view === 'block' && i === 0) || (view === 'chunk' && i === 1)) {{
            btns[i].classList.add('active');
        }}
    }}

    var blockImg = section.querySelector('.block-img');
    var chunkImg = section.querySelector('.chunk-img');
    var blockView = section.querySelector('.block-view');
    var chunkView = section.querySelector('.chunk-view');
    var missingSection = section.querySelector('.missing-blocks-section');

    if (view === 'block') {{
        if (blockImg) blockImg.style.display = 'block';
        if (chunkImg) chunkImg.style.display = 'none';
        var bvs = section.querySelectorAll('.block-view');
        for (var j = 0; j < bvs.length; j++) bvs[j].style.display = 'block';
        if (chunkView) chunkView.style.display = 'none';
    }} else {{
        if (blockImg) blockImg.style.display = 'none';
        if (chunkImg) chunkImg.style.display = 'block';
        var bvs2 = section.querySelectorAll('.block-view');
        for (var k = 0; k < bvs2.length; k++) bvs2[k].style.display = 'none';
        if (chunkView) chunkView.style.display = 'block';
    }}
}}

function onVerdictChange(radio) {{
    var card = radio.closest('.card');
    var textarea = card.querySelector('.correction-input');
    if (radio.value === 'partial' || radio.value === 'wrong') {{
        textarea.style.display = 'block';
    }} else {{
        textarea.style.display = 'none';
    }}
}}

var missingBlockCounter = 0;

function addMissingBlock(pageNum, parserName) {{
    var section = document.querySelector('.page-section[data-parser="' + parserName + '"][data-page="' + pageNum + '"]');
    if (!section) return;

    var select = section.querySelector('.missing-type-select');
    var textInput = section.querySelector('.missing-text-input');
    var blockType = select.value;
    var expectedText = textInput.value.trim();

    if (!expectedText) {{
        alert('Please enter the expected text.');
        return;
    }}

    missingBlockCounter++;
    var list = section.querySelector('.missing-blocks-list');
    var item = document.createElement('div');
    item.className = 'missing-block-item';
    item.dataset.missingId = missingBlockCounter;

    var strong = document.createElement('strong');
    strong.textContent = blockType;
    item.appendChild(strong);

    var preview = document.createTextNode(': ' + expectedText.substring(0, 150) + (expectedText.length > 150 ? '...' : ''));
    item.appendChild(preview);

    item.appendChild(document.createElement('br'));

    var notes = document.createElement('textarea');
    notes.className = 'missing-notes';
    notes.placeholder = 'Notes...';
    notes.style.cssText = 'width:100%;margin-top:4px;min-height:30px;padding:4px;border:1px solid #ddd;border-radius:4px;font-size:12px;';
    item.appendChild(notes);

    var typeInput = document.createElement('input');
    typeInput.type = 'hidden';
    typeInput.className = 'missing-type-value';
    typeInput.value = blockType;
    item.appendChild(typeInput);

    var textInput2 = document.createElement('input');
    textInput2.type = 'hidden';
    textInput2.className = 'missing-text-value';
    textInput2.value = expectedText;
    item.appendChild(textInput2);

    list.appendChild(item);

    textInput.value = '';
}}

function collectAnnotations() {{
    var annotator = document.getElementById('annotatorName').value.trim();
    var docNotes = document.getElementById('docNotes').value.trim();

    var gt = {{
        file_name: REPORT_DATA.file_name,
        file_hash: REPORT_DATA.file_hash,
        parser_name: currentParser,
        annotator: annotator,
        created_at: new Date().toISOString(),
        notes: docNotes,
        pages: []
    }};

    var sections = document.querySelectorAll('.page-section[data-parser="' + currentParser + '"]');
    for (var i = 0; i < sections.length; i++) {{
        var section = sections[i];
        var pageNum = parseInt(section.dataset.page);
        var pageNotes = '';
        var pn = section.querySelector('.page-notes');
        if (pn) pageNotes = pn.value.trim();

        var blockVerdicts = [];
        var cards = section.querySelectorAll('.card[data-block-index]');
        for (var j = 0; j < cards.length; j++) {{
            var card = cards[j];
            var blockIndex = parseInt(card.dataset.blockIndex);
            var blockType = card.dataset.blockType;

            var checked = card.querySelector('input[type="radio"]:checked');
            if (!checked) continue;

            var verdict = checked.value;
            var originalText = '';
            // Find original text from report data
            var parserData = REPORT_DATA.parsers[currentParser];
            if (parserData) {{
                for (var p = 0; p < parserData.pages.length; p++) {{
                    if (parserData.pages[p].page_number === pageNum) {{
                        for (var b = 0; b < parserData.pages[p].blocks.length; b++) {{
                            if (parserData.pages[p].blocks[b].block_index === blockIndex) {{
                                originalText = parserData.pages[p].blocks[b].text;
                                break;
                            }}
                        }}
                        break;
                    }}
                }}
            }}

            var correction = card.querySelector('.correction-input');
            var correctedText = (correction && correction.style.display !== 'none') ? correction.value.trim() : null;

            blockVerdicts.push({{
                block_index: blockIndex,
                block_type: blockType,
                verdict: verdict,
                original_text: originalText,
                corrected_text: correctedText || null,
                notes: null
            }});
        }}

        var missingBlocks = [];
        var missingItems = section.querySelectorAll('.missing-block-item');
        for (var m = 0; m < missingItems.length; m++) {{
            var mi = missingItems[m];
            var mType = mi.querySelector('.missing-type-value').value;
            var mText = mi.querySelector('.missing-text-value').value;
            var mNotes = mi.querySelector('.missing-notes');
            missingBlocks.push({{
                block_type: mType,
                expected_text: mText,
                approximate_position: null,
                notes: mNotes ? mNotes.value.trim() || null : null
            }});
        }}

        gt.pages.push({{
            page_number: pageNum,
            block_verdicts: blockVerdicts,
            missing_blocks: missingBlocks,
            page_notes: pageNotes || null
        }});
    }}

    return gt;
}}

function downloadGroundTruth() {{
    var gt = collectAnnotations();
    var json = JSON.stringify(gt, null, 2);
    var blob = new Blob([json], {{ type: 'application/json' }});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = REPORT_DATA.file_name.replace('.pdf', '') + '.' + currentParser + '.ground_truth.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}}
</script>

</body>
</html>"""

    return html_output
