import React, { useState, useEffect } from 'react';
import { FileText, Download, Copy, RefreshCw, Check, Code, Eye } from 'lucide-react';

export default function ArtifactViewer() {
  const [outputs, setOutputs] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState('rendered'); // 'rendered' | 'raw'

  const fetchOutputs = async () => {
    try {
      const res = await fetch('/api/outputs');
      if (res.ok) {
        const data = await res.json();
        setOutputs(data);
        if (data.length > 0 && !selectedFile) {
          fetchFileContent(data[0].name);
        }
      }
    } catch (err) {
      console.error('Failed to list outputs:', err);
    }
  };

  const fetchFileContent = async (filename) => {
    setSelectedFile(filename);
    setLoading(true);
    try {
      const res = await fetch(`/api/outputs/${filename}`);
      if (res.ok) {
        const text = await res.text();
        setFileContent(text);
      } else {
        setFileContent('Error loading file content.');
      }
    } catch (err) {
      setFileContent(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOutputs();
  }, []);

  const handleCopy = () => {
    navigator.clipboard.writeText(fileContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '1.5rem' }}>
      {/* File List Side Panel */}
      <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div className="card-title">
            <FileText size={18} style={{ color: 'var(--accent-cyan)' }} /> Output Files
          </div>
          <button
            className="btn btn-secondary"
            style={{ padding: '0.3rem 0.5rem' }}
            onClick={fetchOutputs}
            title="Refresh file list"
          >
            <RefreshCw size={14} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '500px', overflowY: 'auto' }}>
          {outputs.length === 0 ? (
            <div style={{ color: 'var(--text-dim)', fontStyle: 'italic', fontSize: '0.85rem', padding: '1rem 0' }}>
              No output files generated yet.
            </div>
          ) : (
            outputs.map((file) => (
              <button
                key={file.name}
                className={`btn ${selectedFile === file.name ? 'btn-primary' : 'btn-secondary'}`}
                style={{
                  justifyContent: 'flex-start',
                  textAlign: 'left',
                  fontSize: '0.85rem',
                  padding: '0.6rem 0.8rem',
                }}
                onClick={() => fetchFileContent(file.name)}
              >
                📄 {file.name}
                <span style={{ marginLeft: 'auto', fontSize: '0.7rem', opacity: 0.7 }}>
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Main File Content Display */}
      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
        <div className="card-header">
          <div className="card-title">
            {selectedFile ? `File: ${selectedFile}` : 'Select a file'}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                className={`btn ${viewMode === 'rendered' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                onClick={() => setViewMode('rendered')}
              >
                <Eye size={14} /> Formatted
              </button>
              <button
                className={`btn ${viewMode === 'raw' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                onClick={() => setViewMode('raw')}
              >
                <Code size={14} /> Raw Markdown
              </button>
            </div>

            <button
              className="btn btn-secondary"
              style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
              onClick={handleCopy}
              disabled={!fileContent}
            >
              {copied ? <Check size={14} style={{ color: '#10b981' }} /> : <Copy size={14} />}
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>

        <div style={{ padding: '1.5rem', flex: 1, overflowY: 'auto', maxHeight: '600px' }}>
          {loading ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '3rem 0' }}>
              Loading output file content...
            </div>
          ) : viewMode === 'raw' ? (
            <pre
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.85rem',
                color: '#a7f3d0',
                whiteSpace: 'pre-wrap',
                margin: 0,
              }}
            >
              {fileContent}
            </pre>
          ) : (
            <div className="markdown-body">
              {fileContent.split('\n\n').map((paragraph, i) => {
                if (paragraph.startsWith('# ')) {
                  return <h1 key={i}>{paragraph.replace('# ', '')}</h1>;
                } else if (paragraph.startsWith('## ')) {
                  return <h2 key={i}>{paragraph.replace('## ', '')}</h2>;
                } else if (paragraph.startsWith('### ')) {
                  return <h3 key={i}>{paragraph.replace('### ', '')}</h3>;
                } else if (paragraph.startsWith('```')) {
                  return (
                    <pre key={i}>
                      <code>{paragraph.replace(/```[a-z]*/g, '').trim()}</code>
                    </pre>
                  );
                } else {
                  return <p key={i}>{paragraph}</p>;
                }
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
