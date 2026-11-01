import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, apiBaseUrl } from './services/api.js'
import { normalizePrediction } from './services/prediction.js'

const MAX_MB = Number(import.meta.env.VITE_MAX_UPLOAD_SIZE_MB) || 10
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']
const COLORS = ['#42c9b2', '#f4ae52', '#64a5ff', '#f0737e', '#bc91ff', '#75c995']
const prettySize = (bytes) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`
function StatusPill({ tone = 'muted', children, pulse = false }) {
  return <span className={`pill ${tone}`}><i className={pulse ? 'pulse' : ''} />{children}</span>
}

function App() {
  const [health, setHealth] = useState(null)
  const [online, setOnline] = useState(false)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [dimensions, setDimensions] = useState(null)
  const [threshold, setThreshold] = useState(0.25)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [showOriginal, setShowOriginal] = useState(false)
  const inputRef = useRef(null)
  const requestRef = useRef(null)

  const refreshHealth = useCallback(async (signal) => {
    try {
      const state = await api.health(signal)
      setHealth(state)
      setOnline(true)
    } catch {
      setOnline(false)
      setHealth(null)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    refreshHealth(controller.signal)
    const poll = window.setInterval(() => refreshHealth(), 15000)
    return () => { controller.abort(); clearInterval(poll); requestRef.current?.abort() }
  }, [refreshHealth])

  useEffect(() => {
    if (!file) { setPreview(''); setDimensions(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    setDimensions(null)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const allCounts = result?.class_counts || {}
  const detectedClasses = useMemo(() => Object.values(allCounts).filter((count) => count > 0).length, [allCounts])
  const sortedDetections = useMemo(() => [...(result?.detections || [])].sort((a, b) => (Number(b?.confidence) || 0) - (Number(a?.confidence) || 0)), [result])

  const chooseFile = (nextFile) => {
    setError('')
    if (!nextFile) return
    if (!ACCEPTED.includes(nextFile.type)) { setError('Choose a JPEG, PNG, or WEBP image.'); return }
    if (nextFile.size > MAX_MB * 1024 * 1024) { setError(`This image is larger than the ${MAX_MB} MB upload limit.`); return }
    setFile(nextFile)
    setResult(null)
    setShowOriginal(false)
  }

  const clearFile = () => {
    setFile(null); setResult(null); setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const analyze = async () => {
    if (!file || busy || !online || !health?.prediction_available) return
    setBusy(true); setError(''); setResult(null)
    const controller = new AbortController()
    requestRef.current = controller
    const timeout = window.setTimeout(() => controller.abort(), 120000)
    try {
      const response = await api.predict(file, threshold, controller.signal)
      setResult(normalizePrediction(response, threshold))
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.')
    } finally {
      clearTimeout(timeout); requestRef.current = null; setBusy(false)
      refreshHealth()
    }
  }

  const imageToShow = result && !showOriginal && result.annotated_image ? result.annotated_image : preview
  const canAnalyze = Boolean(file && online && health?.prediction_available && !busy)
  const statusText = !online ? 'Backend Offline' : health?.prediction_available ? 'Model Ready' : 'Model Missing'
  const statusTone = !online ? 'danger' : health?.prediction_available ? 'good' : 'warn'

  return <main className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><CircuitIcon /></div><div><div className="brand-name">PCB Defect Detection</div><div className="brand-sub">AI-Powered PCB Quality Inspection</div></div></div>
      <div className="top-status"><StatusPill tone={online ? 'good' : 'danger'} pulse={online}>{online ? 'API Connected' : 'Backend Offline'}</StatusPill><StatusPill tone={statusTone}>{statusText}</StatusPill><a className="docs-link" href={`${apiBaseUrl}/docs`} target="_blank" rel="noreferrer">API Docs <span>↗</span></a></div>
    </header>

    <section className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> INSPECTION WORKSPACE</div><h1>PCB quality inspection</h1><p>Analyze board imagery for manufacturing defects with your trained YOLOv8 model.</p></div><div className="heading-meta"><span className="meta-label">MODEL</span><strong>{health?.model_name || 'pcb_defect_best.pt'}</strong><span className="meta-caption">{health?.prediction_available ? 'Custom weights loaded' : online ? 'Weights not detected' : 'Waiting for backend'}</span></div></section>

    {!online && <div className="notice danger-notice"><span className="notice-icon">!</span><div><strong>Backend connection unavailable</strong><p>Start the FastAPI server at {apiBaseUrl}. The connection is checked every 15 seconds.</p></div><button className="text-button" onClick={() => refreshHealth()}>Retry connection</button></div>}
    {online && !health?.prediction_available && <div className="notice warning-notice"><span className="notice-icon">!</span><div><strong>Custom model not ready</strong><p>{health?.model_error || 'The trained model is unavailable.'} {health?.model_error?.toLowerCase().includes('missing') ? <>Place <code>pcb_defect_best.pt</code> in <code>backend/models/</code>, then restart the backend.</> : 'Check the model class order and backend logs.'}</p></div><span className="notice-side">UPLOAD PREVIEW AVAILABLE</span></div>}

    <div className="workspace-grid">
      <section className="panel upload-panel">
        <div className="panel-heading"><div><div className="section-kicker">01 / INPUT</div><h2>Inspection image</h2></div>{file && <button className="icon-button" onClick={clearFile} title="Remove image" aria-label="Remove image">×</button>}</div>
        {!file ? <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragEnter={(e) => { e.preventDefault(); setDragging(true) }} onDragOver={(e) => e.preventDefault()} onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false) }} onDrop={(e) => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files?.[0]) }}>
          <div className="upload-symbol"><UploadIcon /></div><strong>Drop your PCB image here</strong><span>or</span><button className="secondary-button" onClick={() => inputRef.current?.click()}>Browse files</button><small>JPEG · PNG · WEBP <i /> up to {MAX_MB} MB</small>
        </div> : <div className="selected-image">
          <div className="image-frame"><img src={preview} alt="Selected PCB inspection" onLoad={(e) => setDimensions({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })} /><span className="image-tag">SOURCE IMAGE</span></div>
          <div className="file-info"><div className="file-thumb"><ImageIcon /></div><div className="file-data"><strong title={file.name}>{file.name}</strong><span>{prettySize(file.size)}{dimensions && ` · ${dimensions.width} × ${dimensions.height} px`}</span></div><span className="file-check">✓</span></div>
          <button className="replace-link" onClick={() => inputRef.current?.click()}>Replace image</button>
        </div>}
        <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => chooseFile(e.target.files?.[0])} />
        <div className="threshold-control"><div className="threshold-title"><label htmlFor="threshold">Confidence threshold</label><output>{threshold.toFixed(2)}</output></div><input id="threshold" type="range" min="0.05" max="0.95" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} style={{ '--range-progress': `${((threshold - 0.05) / 0.9) * 100}%` }} /><div className="range-ends"><span>More detections</span><span>Higher confidence</span></div></div>
        {error && <div className="inline-error" role="alert"><span>!</span>{error}</div>}
        {online && !health?.prediction_available && <p className="disabled-reason">Analysis is disabled until the custom model is available.</p>}
        <button className="primary-button" disabled={!canAnalyze} onClick={analyze}>{busy ? <><span className="spinner" /> Analyzing image…</> : <><SparkIcon /> Analyze PCB <span className="button-arrow">→</span></>}</button>
      </section>

      <section className="panel results-panel">
        <div className="panel-heading"><div><div className="section-kicker">02 / ANALYSIS</div><h2>Detection results</h2></div>{result && <StatusPill tone="good">Analysis complete</StatusPill>}</div>
        {!result ? <div className="results-empty"><div className="radar"><span /><span /><span /><div><TargetIcon /></div></div><strong>{busy ? 'Inspecting PCB image' : 'Ready for inspection'}</strong><p>{busy ? 'Running custom model inference and preparing annotations.' : 'Upload an image and run analysis to see annotated defects and inspection details.'}</p><div className="empty-steps"><span className={file ? 'step-done' : ''}><i>{file ? '✓' : '1'}</i> Image</span><b /> <span className={busy ? 'step-active' : ''}><i>2</i> Analyze</span><b /> <span><i>3</i> Results</span></div></div> : <>
          <div className="metrics-row"><Metric label="DEFECTS FOUND" value={result.total_defects} accent /><Metric label="CLASSES DETECTED" value={`${detectedClasses} / 6`} /><Metric label="INFERENCE TIME" value={result.processing_time_ms == null ? '—' : `${result.processing_time_ms.toFixed(1)} ms`} /><Metric label="CONFIDENCE" value={`${Math.round(result.confidence_threshold * 100)}%`} /></div>
          <div className="viewer-toolbar"><span>{showOriginal || !result.annotated_image ? 'SOURCE IMAGE' : 'ANNOTATED OUTPUT'}</span><div><button className={`toggle-button ${!showOriginal ? 'active' : ''}`} onClick={() => setShowOriginal(false)}>Annotated</button><button className={`toggle-button ${showOriginal ? 'active' : ''}`} onClick={() => setShowOriginal(true)}>Original</button>{result.annotated_image && <a className="download-button" href={result.annotated_image} download={`${file?.name.replace(/\.[^.]+$/, '') || 'pcb'}-annotated.jpg`} title="Download annotated image"><DownloadIcon /></a>}</div></div>
          <div className="result-image"><img src={imageToShow} alt={showOriginal || !result.annotated_image ? 'Original PCB image' : 'PCB image annotated with detected defects'} />{(result.image_width && result.image_height) || dimensions ? <span className="image-resolution">{result.image_width && result.image_height ? `${result.image_width} × ${result.image_height}` : `${dimensions.width} × ${dimensions.height}`}</span> : null}</div>
          {!result.annotated_image && <p className="disabled-reason">The API response did not include an annotated image.</p>}
          <div className="legend-row">{result.detections.slice(0, 4).map((d, i) => <span key={`${d.class_id ?? d.class_name}-${i}`}><i style={{ background: COLORS[d.class_id % COLORS.length] }} />{d.class_name}</span>)}{result.detections.length > 4 && <span className="legend-more">+{result.detections.length - 4} more</span>}</div>
        </>}
      </section>
    </div>

    {result && <div className="detail-grid">
      <section className="panel distribution-panel"><div className="panel-heading compact"><div><div className="section-kicker">CLASS BREAKDOWN</div><h2>Defect distribution</h2></div><span className="unit-label">DETECTIONS</span></div><div className="distribution-list">{Object.entries(allCounts).map(([name, count], index) => <div className="distribution-item" key={name}><div className="distribution-label"><i style={{ background: COLORS[index % COLORS.length] }} /><span>{name}</span><strong>{count}</strong></div><div className="bar-track"><div className="bar-fill" style={{ width: `${result.total_defects ? (count / result.total_defects) * 100 : 0}%`, background: COLORS[index % COLORS.length] }} /></div></div>)}</div></section>
      <section className="panel detections-panel"><div className="panel-heading compact"><div><div className="section-kicker">DETECTION LOG</div><h2>Individual detections <span className="count-badge">{result.total_defects}</span></h2></div><span className="unit-label">SORTED BY CONFIDENCE</span></div>{sortedDetections.length === 0 ? <div className="no-detections"><div className="check-ring">✓</div><div><strong>No defects detected</strong><p>No objects met the {Math.round(result.confidence_threshold * 100)}% confidence threshold.</p></div></div> : <div className="table-scroll"><table><thead><tr><th>#</th><th>DEFECT CLASS</th><th>CONFIDENCE</th><th>BOUNDING BOX <small>· px</small></th></tr></thead><tbody>{sortedDetections.map((d, index) => <tr key={`${d.class_id}-${index}`}><td className="row-index">{String(index + 1).padStart(2, '0')}</td><td><span className="class-chip"><i style={{ background: COLORS[(d.class_id ?? 0) % COLORS.length] }} />{d.class_name}</span></td><td><span className="confidence-value">{Number.isFinite(d.confidence) ? `${(d.confidence * 100).toFixed(1)}%` : '—'}</span></td><td className="coordinates">{d.bbox ? `[${d.bbox.x1}, ${d.bbox.y1}] – [${d.bbox.x2}, ${d.bbox.y2}]` : '—'}</td></tr>)}</tbody></table></div>}</section>
    </div>}
    <footer><span><i /> LOCAL INFERENCE</span><span>PCB DEFECT DETECTION <b>·</b> YOLOv8</span></footer>
  </main>
}

function Metric({ label, value, accent }) { return <div className="metric"><span>{label}</span><strong className={accent ? 'metric-accent' : ''}>{value}</strong></div> }
function CircuitIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3v4M16 3v4M8 17v4M16 17v4M3 8h4M17 8h4M3 16h4M17 16h4"/><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M10 10h4v4h-4z"/></svg> }
function UploadIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15V3m0 0L7.5 7.5M12 3l4.5 4.5M4 15v4a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg> }
function ImageIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg> }
function TargetIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><path d="M12 1v3M23 12h-3M12 23v-3M1 12h3"/></svg> }
function DownloadIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 17v3h16v-3"/></svg> }
function SparkIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3zM19 16l.9 2.1L22 19l-2.1.9L19 22l-.9-2.1L16 19l2.1-.9L19 16z"/></svg> }

export default App
