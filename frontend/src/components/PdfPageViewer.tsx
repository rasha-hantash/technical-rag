import { useState, useRef, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { BboxHighlight } from './BboxHighlight'
import { getPdfFileUrl } from '../lib/api'

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl

interface PdfPageViewerProps {
  documentId: string
  pageNumber: number
  bbox: [number, number, number, number] | null
}

export function PdfPageViewer({
  documentId,
  pageNumber,
  bbox,
}: PdfPageViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)
  const [scaleFactor, setScaleFactor] = useState(1)

  useEffect(() => {
    if (!containerRef.current) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width)
      }
    })

    observer.observe(containerRef.current)
    setContainerWidth(containerRef.current.clientWidth)

    return () => observer.disconnect()
  }, [])

  const handlePageLoad = (page: { originalWidth: number }) => {
    if (containerWidth > 0 && page.originalWidth > 0) {
      setScaleFactor(containerWidth / page.originalWidth)
    }
  }

  return (
    <div ref={containerRef} className="w-full">
      {containerWidth > 0 && (
        <Document
          file={getPdfFileUrl(documentId)}
          loading={
            <div className="flex items-center justify-center py-12 text-sm text-stone-400">
              Loading PDF...
            </div>
          }
          error={
            <div className="flex items-center justify-center py-12 text-sm text-red-400">
              Failed to load PDF
            </div>
          }
        >
          <div className="relative inline-block">
            <Page
              pageNumber={pageNumber}
              width={containerWidth}
              onLoadSuccess={handlePageLoad}
              renderTextLayer={false}
              renderAnnotationLayer={false}
            />
            {bbox && <BboxHighlight bbox={bbox} scaleFactor={scaleFactor} />}
          </div>
        </Document>
      )}

      {!bbox && (
        <p className="mt-1 text-center text-xs text-stone-400 italic">
          Exact highlight location unavailable
        </p>
      )}
    </div>
  )
}
