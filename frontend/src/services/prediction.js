/** Preserve the API's detections and fill optional presentation metadata safely. */
export function createPredictionForm(file, threshold) {
  const form = new FormData()
  form.append('file', file)
  form.append('confidence_threshold', String(threshold))
  return form
}

export function normalizePrediction(payload, requestedThreshold) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('The backend returned an invalid prediction response.')
  }

  const detections = Array.isArray(payload.detections) ? payload.detections : []
  const suppliedCounts = payload.class_counts
  const classCounts = suppliedCounts && typeof suppliedCounts === 'object' && !Array.isArray(suppliedCounts)
    ? suppliedCounts
    : detections.reduce((counts, detection) => {
      const className = detection?.class_name
      if (typeof className === 'string') counts[className] = (counts[className] || 0) + 1
      return counts
    }, {})

  return {
    ...payload,
    total_defects: Number.isFinite(payload.total_defects) ? payload.total_defects : detections.length,
    class_counts: classCounts,
    detections,
    confidence_threshold: Number.isFinite(payload.confidence_threshold)
      ? payload.confidence_threshold
      : requestedThreshold,
    processing_time_ms: Number.isFinite(payload.processing_time_ms) ? payload.processing_time_ms : null,
    image_width: Number.isFinite(payload.image_width) ? payload.image_width : null,
    image_height: Number.isFinite(payload.image_height) ? payload.image_height : null,
    annotated_image: typeof payload.annotated_image === 'string' && payload.annotated_image.startsWith('data:image/')
      ? payload.annotated_image
      : null,
  }
}
