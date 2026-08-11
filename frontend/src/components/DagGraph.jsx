import React from 'react';
import { ArrowRight, CheckCircle2, Clock, AlertTriangle, Play, CornerDownRight, ShieldCheck } from 'lucide-react';

export default function DagGraph({ workflow, nodeStatuses = {}, activeNodeId }) {
  if (!workflow || !workflow.nodes) {
    return (
      <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No workflow selected.
      </div>
    );
  }

  // Calculate execution progress
  const totalNodes = workflow.nodes.length;
  const completedNodes = Object.values(nodeStatuses).filter(
    (s) => s.status === 'success' || s.status === 'failed' || s.status === 'skipped'
  ).length;
  const progressPercent = totalNodes > 0 ? Math.round((completedNodes / totalNodes) * 100) : 0;

  return (
    <div className="glass-panel" style={{ overflow: 'hidden' }}>
      <div className="card-header">
        <div>
          <div className="card-title">
            Workflow DAG: <span style={{ color: 'var(--accent-cyan)' }}>{workflow.name}</span>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            {workflow.description}
          </div>
        </div>

        {/* Progress Pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>PROGRESS</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>
              {completedNodes} / {totalNodes} Nodes ({progressPercent}%)
            </div>
          </div>
          <div
            style={{
              width: '120px',
              height: '8px',
              background: 'rgba(255, 255, 255, 0.08)',
              borderRadius: '999px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${progressPercent}%`,
                height: '100%',
                background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-purple))',
                transition: 'width 0.4s ease',
              }}
            />
          </div>
        </div>
      </div>

      <div className="dag-container">
        {workflow.nodes.map((node, index) => {
          const statusInfo = nodeStatuses[node.id] || { status: 'pending' };
          const status = statusInfo.status;
          const isRunning = activeNodeId === node.id || status === 'running';

          let statusColor = 'var(--status-pending)';
          let statusIcon = <Clock size={16} />;
          let cardClass = 'pending';

          if (isRunning) {
            statusColor = 'var(--status-running)';
            statusIcon = <Play size={16} style={{ animation: 'spin 2s linear infinite' }} />;
            cardClass = 'running';
          } else if (status === 'success') {
            statusColor = 'var(--status-success)';
            statusIcon = <CheckCircle2 size={16} />;
            cardClass = 'success';
          } else if (status === 'failed') {
            statusColor = 'var(--status-failed)';
            statusIcon = <AlertTriangle size={16} />;
            cardClass = 'failed';
          } else if (status === 'skipped') {
            statusColor = 'var(--status-skipped)';
            statusIcon = <AlertTriangle size={16} />;
            cardClass = 'pending';
          }

          return (
            <React.Fragment key={node.id}>
              {index > 0 && node.inputs && node.inputs.length > 0 && (
                <div className="dag-arrow">
                  <ArrowRight size={24} />
                </div>
              )}

              <div className={`dag-node-card ${cardClass}`}>
                <div className="dag-node-header">
                  <div className="dag-node-id">{node.id}</div>
                  <div className="dag-node-role">{node.role}</div>
                </div>

                <div className="dag-node-goal" title={node.goal}>
                  {node.goal}
                </div>

                {node.inputs && node.inputs.length > 0 && (
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <CornerDownRight size={12} /> Depends on: <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>{node.inputs.join(', ')}</span>
                  </div>
                )}

                <div className="dag-node-tools">
                  {node.tools && node.tools.length > 0 ? (
                    node.tools.map((t) => (
                      <span key={t} className="tool-chip">
                        🛠️ {t}
                        {node.required_tools && node.required_tools.includes(t) && (
                          <ShieldCheck size={10} style={{ marginLeft: '4px', color: '#10b981', display: 'inline' }} />
                        )}
                      </span>
                    ))
                  ) : (
                    <span className="tool-chip" style={{ opacity: 0.5 }}>No tools</span>
                  )}
                </div>

                {/* Status Footer Badge */}
                <div
                  style={{
                    marginTop: '1rem',
                    paddingTop: '0.6rem',
                    borderTop: '1px solid var(--border-subtle)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.75rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: statusColor, fontWeight: 700 }}>
                    {statusIcon}
                    <span style={{ textTransform: 'uppercase' }}>{status}</span>
                  </div>
                  <div style={{ color: 'var(--text-dim)' }}>
                    Max iter: {node.max_iterations || 6}
                  </div>
                </div>
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
