import React, { useState, useEffect, useRef } from 'react';
import { Play, Terminal, Zap, CheckCircle2, AlertCircle, Layers, Wrench, FileCode } from 'lucide-react';

const SAMPLE_PROMPTS = [
  "Research the state of solid-state batteries in 2026 and summarize market outlook and key commercialization milestones.",
  "Analyze AI agent orchestration architectures (DAG vs Chain) and produce a detailed comparison matrix.",
  "Investigate quantum computing milestones in supply chain optimization and summarize recent breakthroughs.",
];

export default function ExecutionConsole({
  workflows,
  selectedWorkflow,
  setSelectedWorkflow,
  onRunStart,
  runStatus,
  logs,
  setLogs,
  nodeStatuses,
  setNodeStatuses,
  activeNodeId,
  setActiveNodeId,
  finalOutput,
  setFinalOutput,
}) {
  const [taskPrompt, setTaskPrompt] = useState(
    "Research the state of solid-state batteries and summarize the market outlook"
  );
  const [mockMode, setMockMode] = useState(true);
  const [selectedModel, setSelectedModel] = useState("llama-3.3-70b-versatile");
  const consoleEndRef = useRef(null);

  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleRun = () => {
    if (!selectedWorkflow) return;
    setLogs([]);
    setNodeStatuses({});
    setFinalOutput(null);

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/api/ws/run`;
    const ws = new WebSocket(wsUrl);
    let workflowFinished = false;

    onRunStart('running');

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          workflow_name: selectedWorkflow.name,
          task_prompt: taskPrompt,
          mock: mockMode,
          model: selectedModel,
        })
      );
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const timestamp = new Date().toLocaleTimeString();

      if (msg.type === 'workflow_start') {
        setLogs((prev) => [
          ...prev,
          {
            time: timestamp,
            type: 'system',
            text: `🚀 Started workflow '${msg.workflow_name}' (${msg.mock ? 'Mock Mode' : 'Groq API'})`,
          },
        ]);
      } else if (msg.type === 'node_start') {
        if (setActiveNodeId) setActiveNodeId(msg.node_id);
        setNodeStatuses((prev) => ({
          ...prev,
          [msg.node_id]: { status: 'running' },
        }));
        setLogs((prev) => [
          ...prev,
          {
            time: timestamp,
            type: 'node',
            node: msg.node_id,
            text: `⚡ Agent Node '${msg.node_id}' started execution...`,
          },
        ]);
      } else if (msg.type === 'tool_call') {
        setLogs((prev) => [
          ...prev,
          {
            time: timestamp,
            type: 'tool',
            node: msg.node_id,
            tool: msg.tool.name,
            arguments: msg.tool.arguments,
            result: msg.tool.result,
            error: msg.tool.error,
            text: `🛠️ Tool '${msg.tool.name}' called by '${msg.node_id}'`,
          },
        ]);
      } else if (msg.type === 'node_complete') {
        if (setActiveNodeId) setActiveNodeId(null);
        const res = msg.result;
        const nodeSucceeded = res.status === 'success';
        const nodeSkipped = res.status === 'skipped';
        setNodeStatuses((prev) => ({
          ...prev,
          [msg.node_id]: { status: res.status, text: res.text, artifacts: res.artifacts, error: res.error },
        }));
        setLogs((prev) => [
          ...prev,
          {
            time: timestamp,
            type: nodeSucceeded ? 'success' : nodeSkipped ? 'warning' : 'error',
            node: msg.node_id,
            text: `${nodeSucceeded ? '[ok]' : nodeSkipped ? '[skipped]' : '[failed]'} Node '${msg.node_id}' finished with status [${res.status}]`,
            output: res.text || res.error,
          },
        ]);
      } else if (msg.type === 'workflow_complete') {
        if (setActiveNodeId) setActiveNodeId(null);
        workflowFinished = true;
        const workflowSucceeded = msg.status === 'completed';
        onRunStart(workflowSucceeded ? 'completed' : 'failed');
        setFinalOutput({
          final_node_id: msg.final_node_id,
          text: msg.final_text,
          status: msg.status,
          failed_nodes: msg.failed_nodes || [],
          skipped_nodes: msg.skipped_nodes || [],
        });
        if (!workflowSucceeded) {
          setLogs((prev) => [
            ...prev,
            {
              time: timestamp,
              type: 'error',
              text: `Workflow failed. Failed nodes: ${(msg.failed_nodes || []).join(', ') || 'none'}; skipped nodes: ${(msg.skipped_nodes || []).join(', ') || 'none'}.`,
            },
          ]);
          ws.close();
          return;
        }
        setLogs((prev) => [
          ...prev,
          {
            time: timestamp,
            type: workflowSucceeded ? 'complete' : 'error',
            text: `🎉 Workflow execution completed successfully! Final output node: ${msg.final_node_id}`,
          },
        ]);
        ws.close();
      } else if (msg.type === 'error') {
        if (setActiveNodeId) setActiveNodeId(null);
        workflowFinished = true;
        onRunStart('failed');
        setLogs((prev) => [
          ...prev,
          {
            time: timestamp,
            type: 'error',
            text: `❌ Error: ${msg.message}`,
          },
        ]);
        ws.close();
      }
    };

    ws.onerror = (err) => {
      onRunStart('failed');
      setLogs((prev) => [
        ...prev,
        {
          time: new Date().toLocaleTimeString(),
          type: 'error',
          text: `WebSocket error encountered. Make sure the FastAPI server is running and the Vite proxy is pointing at it.`,
        },
      ]);
    };

    ws.onclose = (event) => {
      if (!workflowFinished && !event.wasClean) {
        onRunStart('failed');
        setLogs((prev) => [
          ...prev,
          {
            time: new Date().toLocaleTimeString(),
            type: 'error',
            text: `Workflow connection closed unexpectedly (${event.code || 'unknown close code'}).`,
          },
        ]);
      }
    };
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
      {/* Run Configuration Form */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div className="card-title" style={{ fontSize: '1.1rem' }}>
          <Zap size={20} style={{ color: 'var(--accent-cyan)' }} /> Workflow Controls & Task Input
        </div>

        <div className="input-group">
          <label className="input-label">Select Workflow</label>
          <select
            className="select-input"
            value={selectedWorkflow ? selectedWorkflow.name : ''}
            onChange={(e) => {
              const wf = workflows.find((w) => w.name === e.target.value);
              if (wf) setSelectedWorkflow(wf);
            }}
          >
            {workflows.map((wf) => (
              <option key={wf.name} value={wf.name}>
                {wf.name} ({wf.nodes ? wf.nodes.length : 0} nodes)
              </option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label className="input-label">Task Instructions Prompt</label>
          <textarea
            className="textarea-input"
            rows={4}
            value={taskPrompt}
            onChange={(e) => setTaskPrompt(e.target.value)}
            placeholder="Type your prompt for the multi-agent graph..."
          />
        </div>

        {/* Preset Prompt Chips */}
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: '0.4rem', fontWeight: 600 }}>
            QUICK PRESETS:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {SAMPLE_PROMPTS.map((sample, i) => (
              <button
                key={i}
                type="button"
                className="btn btn-secondary"
                style={{ justifyContent: 'flex-start', fontSize: '0.75rem', padding: '0.4rem 0.75rem', textAlign: 'left' }}
                onClick={() => setTaskPrompt(sample)}
              >
                💡 {sample}
              </button>
            ))}
          </div>
        </div>

        {/* Mode & Options */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
          <div className="input-group">
            <label className="input-label">Execution Mode</label>
            <button
              type="button"
              className={`btn ${mockMode ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setMockMode(!mockMode)}
              style={{ width: '100%', fontSize: '0.8rem' }}
            >
              {mockMode ? '⚡ Mock LLM (Offline)' : '🌐 Groq API (Live)'}
            </button>
          </div>

          <div className="input-group">
            <label className="input-label">Groq Model</label>
            <select
              className="select-input"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={mockMode}
            >
              <option value="llama-3.3-70b-versatile">llama-3.3-70b-versatile</option>
              <option value="llama3-8b-8192">llama3-8b-8192</option>
              <option value="mixtral-8x7b-32768">mixtral-8x7b-32768</option>
            </select>
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleRun}
          disabled={runStatus === 'running' || !selectedWorkflow}
          style={{ marginTop: '0.5rem', padding: '0.9rem', fontSize: '1rem' }}
        >
          <Play size={18} />
          {runStatus === 'running' ? 'Executing Workflow Graph...' : 'Execute Workflow'}
        </button>
      </div>

      {/* Terminal Live Console */}
      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
        <div className="card-header">
          <div className="card-title">
            <Terminal size={18} style={{ color: 'var(--accent-cyan)' }} /> Execution Live Trace Stream
          </div>
          <button
            className="btn btn-secondary"
            style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
            onClick={() => setLogs([])}
          >
            Clear Console
          </button>
        </div>

        <div style={{ padding: '1.25rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="console-box">
            {logs.length === 0 ? (
              <div style={{ color: 'var(--text-dim)', fontStyle: 'italic', textAlign: 'center', padding: '3rem 0' }}>
                Console ready. Click "Execute Workflow" to stream agent events live.
              </div>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="log-entry">
                  <span className="log-time">[{log.time}]</span>
                  {log.node && <span className="log-node">{log.node}</span>}
                  <div style={{ flex: 1 }}>
                    <div>{log.text}</div>
                    {log.type === 'tool' && (
                      <div
                        style={{
                          marginTop: '4px',
                          padding: '6px 10px',
                          background: 'rgba(255, 255, 255, 0.03)',
                          borderRadius: '4px',
                          border: '1px solid var(--border-subtle)',
                          fontSize: '0.78rem',
                          color: '#cbd5e1',
                        }}
                      >
                        <div style={{ fontWeight: 600, color: '#f472b6' }}>
                          Args: {JSON.stringify(log.arguments)}
                        </div>
                        {log.result && <div style={{ marginTop: '2px' }}>Result: {log.result}</div>}
                        {log.error && <div style={{ color: '#ef4444', marginTop: '2px' }}>Error: {log.error}</div>}
                      </div>
                    )}
                    {log.output && (
                      <div
                        style={{
                          marginTop: '6px',
                          padding: '8px',
                          background: 'rgba(0, 0, 0, 0.4)',
                          borderRadius: '4px',
                          fontSize: '0.78rem',
                          color: '#e2e8f0',
                          whiteSpace: 'pre-wrap',
                          maxHeight: '120px',
                          overflowY: 'auto',
                        }}
                      >
                        {log.output}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={consoleEndRef} />
          </div>

          {/* Final Result Card Banner */}
          {finalOutput && (
            <div
              style={{
                marginTop: '1rem',
                padding: '1rem',
                background: finalOutput.status === 'failed' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                border: `1px solid ${finalOutput.status === 'failed' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
                borderRadius: '8px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: finalOutput.status === 'failed' ? '#ef4444' : '#10b981', fontWeight: 700, marginBottom: '0.4rem' }}>
                {finalOutput.status === 'failed' ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}
                {finalOutput.status === 'failed'
                  ? `Workflow failed (${finalOutput.failed_nodes?.length || 0} failed, ${finalOutput.skipped_nodes?.length || 0} skipped)`
                  : `Final Output Delivered by Agent '${finalOutput.final_node_id}'`}
              </div>
              <div style={{ fontSize: '0.85rem', color: '#e2e8f0', whiteSpace: 'pre-wrap', maxHeight: '180px', overflowY: 'auto' }}>
                {finalOutput.text || 'Check the live trace above for node error details.'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
