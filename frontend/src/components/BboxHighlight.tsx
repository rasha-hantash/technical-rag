interface BboxHighlightProps {
  bbox: [number, number, number, number]
  scaleFactor: number
}

export function BboxHighlight({ bbox, scaleFactor }: BboxHighlightProps) {
  const [x0, y0, x1, y1] = bbox

  return (
    <div
      style={{
        position: 'absolute',
        left: x0 * scaleFactor,
        top: y0 * scaleFactor,
        width: (x1 - x0) * scaleFactor,
        height: (y1 - y0) * scaleFactor,
        backgroundColor: 'rgba(218, 119, 86, 0.25)',
        border: '2px solid rgba(218, 119, 86, 0.55)',
        borderRadius: 2,
        pointerEvents: 'none',
        zIndex: 10,
      }}
    />
  )
}
