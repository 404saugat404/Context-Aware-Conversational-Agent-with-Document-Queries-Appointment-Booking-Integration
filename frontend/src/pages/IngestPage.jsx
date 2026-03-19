import { useState, useRef } from 'react'
import { ingestFile } from '../api/client'
import './IngestPage.css'

const ACCEPTED = '.txt,.md,.csv,.json'

function IngestPage() {
  const [file, setFile] = useState(null)
  const [chunkSize, setChunkSize] = useState(512)
  const [chunkOverlap, setChunkOverlap] = useState(64)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleFile = (f) => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files?.[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    setDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const data = await ingestFile(file, chunkSize, chunkOverlap)
      setResult(data)
      setFile(null)
      if (inputRef.current) inputRef.current.value = ''
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="ingest-page">
      <div className="ingest-container">
        <div className="ingest-header">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="12" y1="18" x2="12" y2="12" />
            <line x1="9" y1="15" x2="15" y2="15" />
          </svg>
          <div>
            <h2>Upload Documents</h2>
            <p>Upload files to the knowledge base for RAG queries.</p>
          </div>
        </div>

        <form className="ingest-form" onSubmit={handleSubmit}>
          <div
            className={`drop-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              hidden
            />
            {file ? (
              <div className="file-info">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                <span className="file-name">{file.name}</span>
                <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
            ) : (
              <div className="drop-prompt">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span>Drag & drop a file here, or click to browse</span>
                <span className="drop-hint">Supported: .txt, .md, .csv, .json</span>
              </div>
            )}
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="chunk_size">Chunk Size</label>
              <input
                id="chunk_size"
                type="number"
                min={100}
                max={2048}
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
              />
              <span className="form-hint">100 – 2048 characters</span>
            </div>

            <div className="form-group">
              <label htmlFor="chunk_overlap">Chunk Overlap</label>
              <input
                id="chunk_overlap"
                type="number"
                min={0}
                max={256}
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
              />
              <span className="form-hint">0 – 256 characters</span>
            </div>
          </div>

          <button className="btn-upload" type="submit" disabled={!file || loading}>
            {loading ? 'Uploading...' : 'Upload & Ingest'}
          </button>
        </form>

        {result && (
          <div className="result-card success">
            <p className="result-message">{result.message}</p>
            {result.chunks_ingested != null && (
              <p className="result-detail">Chunks created: <strong>{result.chunks_ingested}</strong></p>
            )}
          </div>
        )}

        {error && (
          <div className="result-card fail">
            <p className="result-message">{error}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default IngestPage
