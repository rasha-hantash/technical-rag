# Plan: Full PDF Reader with Citation Navigation + In-Document Search

## Context

The RAG system returns answers with citations (book title, section, page, bbox). Currently, expanding an evidence card shows a single-page PDF render of the cited page. The user wants to go deeper: open the full book, scroll around the cited area, and search within the PDF for related content the RAG might not have retrieved.

**What exists today:**
- `PdfPageViewer.tsx` renders ONE page via react-pdf v9 + pdfjs-dist 4.8.69 with bbox highlighting
- `EvidenceCard.tsx` embeds `PdfPageViewer` when expanded
- `EvidencePanel.tsx` lists source cards in a scrollable right panel (50% width)
- Layout: `index.tsx` renders `ChatPanel` (left) + `EvidencePanel` (right, shown on query)
- Backend: `GET /api/v1/documents/{id}/file` serves full PDFs (300-1300 pages)
- `BboxHighlight.tsx` overlays a terracotta rectangle at bbox coordinates

**What the user wants:**
1. Click a citation and open the full book scrolled to that page with bbox highlight
2. Scroll through the rest of the book around the cited area
3. Search within the full PDF for terms beyond what RAG retrieved

---

## Approach: Tabbed Right Panel (Sources | Reader)

Replace direct `EvidencePanel` usage with a `RightPanel` wrapper that has two tabs:
- **Sources** tab — existing evidence card list (unchanged)
- **Reader** tab — full scrollable PDF viewer with toolbar + search

When the user clicks "Open in Reader" on an evidence card, the panel switches to the Reader tab, loads the book, and scrolls to the cited page with bbox highlight. A back button returns to Sources.

**Why this layout:** A third panel would make the 50/50 split too cramped. An overlay loses chat context. Tabs keep the spatial layout unchanged and the transition is one click.

---

## New Components

### `frontend/src/components/RightPanel.tsx`
Tab wrapper. Manages `activeTab: 'sources' | 'reader'` and `readerTarget` state. Renders either `EvidencePanel` or `PdfReaderPanel` based on active tab.

### `frontend/src/components/pdf-reader/PdfReaderPanel.tsx`
Main reader container. Props: `documentId`, `bookTitle`, `initialPage`, `initialBbox`, `onClose`.
- Loads PDF via react-pdf `<Document>`
- Stores `numPages` on load
- Delegates to toolbar, search, and content sub-components

### `frontend/src/components/pdf-reader/PdfReaderToolbar.tsx`
Horizontal bar at top: back button, book title, "Page X of Y" with input, prev/next page, zoom (fit-width / 100% / 150%), search toggle.

### `frontend/src/components/pdf-reader/PdfReaderContent.tsx`
Scrollable multi-page container with virtualization:
- Renders placeholder divs for all pages with pre-calculated heights (aspect ratio from page 1)
- `IntersectionObserver` detects which pages are near the viewport
- Only mounts react-pdf `<Page>` for pages in buffer (current +/- 3 pages)
- Pages outside buffer show grey placeholder with page number
- `scrollToPage(n)` scrolls a page wrapper into view
- Tracks current page via intersection ratios (for toolbar display)
- `renderTextLayer={true}` on mounted pages (needed for search)

### `frontend/src/components/pdf-reader/PdfVirtualPage.tsx`
Single page wrapper: renders `<Page>` with text layer + optional `BboxHighlight` overlay + optional search highlights.

### `frontend/src/components/pdf-reader/PdfReaderSearch.tsx`
Search bar below toolbar (toggled):
- Text input with debounced search (300ms)
- Extracts text via `page.getTextContent()` lazily (batches of 50 pages via `requestIdleCallback`), cached in `Map<number, string>`
- Shows "N of M matches", next/prev buttons
- Highlights matching text spans in the text layer via CSS class

### `frontend/src/hooks/usePdfSearch.ts`
Encapsulates search logic: text cache, `search(query)`, `nextMatch()`, `prevMatch()`, `clearSearch()`. Returns `{ matches, currentMatchIndex, isSearching }`.

---

## Modified Files

### `frontend/src/components/EvidenceCard.tsx`
Add "Open in Reader" button in the expanded view. New prop: `onOpenReader?: (target: ReaderTarget) => void`.

### `frontend/src/components/EvidencePanel.tsx`
Accept + pass `onOpenReader` callback to each `EvidenceCard`.

### `frontend/src/routes/index.tsx`
- Replace `<EvidencePanel>` with `<RightPanel>`
- Add state: `rightTab`, `readerTarget`
- Add `handleOpenReader` callback: sets target + switches tab

### `frontend/src/lib/types.ts`
Add `ReaderTarget` interface: `{ documentId, bookTitle?, page?, bbox? }`.

### `frontend/src/components/PdfPageViewer.tsx`
No changes — stays for single-page previews in evidence cards.

---

## Virtualization Strategy (300-1300 page PDFs)

1. On document load, get page 1 viewport → derive aspect ratio
2. Assume all pages share that ratio (true for 99% of technical books)
3. Total scroll height = `numPages * (containerWidth / pageWidth * pageHeight)`
4. Mount `<Page>` only for pages in viewport +/- 3 page buffer (~7-9 pages rendered at any time)
5. If a page has a different aspect ratio on render, adjust its placeholder height

---

## Graphite Stack

```
main
 └─ feat: add RightPanel with tab switching (Sources | Reader)
     └─ feat: multi-page PDF reader with virtualized scrolling + citation jump
         └─ feat: in-PDF text search with highlight
```

### Diff 1: RightPanel shell
Create `RightPanel.tsx`, move `EvidencePanel` under Sources tab, Reader tab shows placeholder. Modify `index.tsx`. Add `ReaderTarget` type. Zero behavior change — purely structural.

### Diff 2: Multi-page reader with citation navigation
Create `PdfReaderPanel`, `PdfReaderContent`, `PdfVirtualPage`, `PdfReaderToolbar`. Add "Open in Reader" to `EvidenceCard`. Wire citation click → reader tab switch → scroll to page + bbox highlight.

### Diff 3: In-PDF text search
Create `usePdfSearch` hook + `PdfReaderSearch` component. Enable text layer on virtual pages. Add search highlight CSS.

---

## Intentionally Excluded

- **Page thumbnails sidebar** — section tree from `/documents/{id}/sections` could serve this later
- **Zoom pinch gestures** — button zoom sufficient for desktop
- **PDF annotations/notes** — out of scope
- **Cross-document search** — already handled by the RAG pipeline
- **New PDF library** — `@react-pdf-viewer/core` bundles its own pdfjs-dist (version conflict). Building on existing react-pdf v9 is cleaner for this scope.

---

## Verification

1. Open a 1300-page book (Programming Rust) — confirm smooth scrolling, only ~7 pages rendered in DOM at any time
2. Click citation in evidence card → Reader opens at correct page with bbox highlight
3. Search for "ownership" → matches highlighted in text layer, next/prev navigates between them
4. Back button returns to Sources tab with state preserved
5. `npx tsc --noEmit` passes
