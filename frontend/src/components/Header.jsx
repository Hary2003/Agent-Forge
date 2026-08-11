import React from 'react';
import { Cpu, PlayCircle, Edit3, FileText, Activity } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, serverConnected }) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-icon">
          <Cpu size={22} />
        </div>
        <div>
          <div className="brand-title">AGENT-FORGE</div>
          <div className="brand-subtitle">DAG Multi-Agent Task Orchestrator</div>
        </div>
      </div>

      <nav className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'orchestrator' ? 'active' : ''}`}
          onClick={() => setActiveTab('orchestrator')}
        >
          <PlayCircle size={16} /> Run & Console
        </button>
        <button
          className={`tab-btn ${activeTab === 'builder' ? 'active' : ''}`}
          onClick={() => setActiveTab('builder')}
        >
          <Edit3 size={16} /> Workflow Builder
        </button>
        <button
          className={`tab-btn ${activeTab === 'artifacts' ? 'active' : ''}`}
          onClick={() => setActiveTab('artifacts')}
        >
          <FileText size={16} /> Output Artifacts
        </button>
      </nav>

      <div className="status-badge" style={{
        backgroundColor: serverConnected ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
        color: serverConnected ? '#10b981' : '#ef4444',
        borderColor: serverConnected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)',
      }}>
        <div
          className="status-dot"
          style={{
            backgroundColor: serverConnected ? '#10b981' : '#ef4444',
            boxShadow: serverConnected ? '0 0 8px #10b981' : '0 0 8px #ef4444',
            animation: serverConnected ? 'pulse-green 2s infinite' : 'none',
          }}
        />
        {serverConnected ? 'API Server Online' : 'API Connecting...'}
      </div>
    </header>
  );
}
