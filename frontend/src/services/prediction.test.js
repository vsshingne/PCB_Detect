import test from 'node:test'
import assert from 'node:assert/strict'
import { createPredictionForm, normalizePrediction } from './prediction.js'

test('builds the multipart request with the selected file and confidence threshold', () => {
  const file = new Blob(['board image'], { type: 'image/jpeg' })
  const form = createPredictionForm(file, 0.25)

  assert.equal(form.get('file').type, 'image/jpeg')
  assert.equal(form.get('confidence_threshold'), '0.25')
})

const detections = [
  { class_id: 3, class_name: 'Short', confidence: 0.93, bbox: { x1: 10, y1: 12, x2: 30, y2: 38 } },
  { class_id: 3, class_name: 'Short', confidence: 0.81, bbox: { x1: 42, y1: 11, x2: 62, y2: 40 } },
  { class_id: 3, class_name: 'Short', confidence: 0.74, bbox: { x1: 71, y1: 20, x2: 94, y2: 45 } },
]

test('keeps the API prediction fields and every detection unchanged', () => {
  const response = {
    total_defects: 3,
    class_counts: { Short: 3 },
    detections,
    annotated_image: 'data:image/jpeg;base64,dGVzdA==',
  }

  const result = normalizePrediction(response, 0.25)

  assert.equal(result.total_defects, 3)
  assert.deepEqual(result.class_counts, { Short: 3 })
  assert.deepEqual(result.detections, detections)
  assert.equal(result.detections.length, 3)
  assert.equal(result.annotated_image, response.annotated_image)
  assert.equal(result.confidence_threshold, 0.25)
  assert.equal(result.processing_time_ms, null)
})

test('preserves zero detections and zero class counts', () => {
  const result = normalizePrediction({
    total_defects: 0,
    class_counts: { Short: 0 },
    detections: [],
    annotated_image: 'data:image/jpeg;base64,empty',
  }, 0.25)

  assert.equal(result.total_defects, 0)
  assert.deepEqual(result.class_counts, { Short: 0 })
  assert.deepEqual(result.detections, [])
  assert.equal(result.annotated_image, 'data:image/jpeg;base64,empty')
})

test('derives only missing display counts and safely handles optional image metadata', () => {
  const result = normalizePrediction({ detections, annotated_image: 'invalid' }, 0.4)

  assert.deepEqual(result.class_counts, { Short: 3 })
  assert.equal(result.total_defects, 3)
  assert.equal(result.confidence_threshold, 0.4)
  assert.equal(result.image_width, null)
  assert.equal(result.image_height, null)
  assert.equal(result.annotated_image, null)
})
