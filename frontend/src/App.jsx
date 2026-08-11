import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import DagGraph from './components/DagGraph';
import ExecutionConsole from './components/ExecutionConsole';
import WorkflowEditor from './components/WorkflowEditor';
import ArtifactViewer from './components/ArtifactViewer';

export default function App() {
  const [activeTab, setActiveTab] = useState('orchestrator');
  const [serverConnected, setServerConnected] = useState(false);
  const [workflows, setWorkflows] = useState([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState(null);
  const [availableTools, setAvailableTools] = useState([]);

  // Execution states
  const [runStatus, setRunStatus] = useState('idle'); // 'idle' | 'running' | 'completed' | 'failed'
  const [nodeStatuses, setNodeStatuses] = useState({});
  const [activeNodeId, setActiveNodeId] = useState(null);
  const [logs, setLogs] = useState([]);
  const [finalOutput, setFinalOutput] = useState(null);

  const fetchInitialData = async () => {
    try {
      const healthRes = await fetch('/api/health');
      if (healthRes.ok) {
        setServerConnected(true);
      } else {
        setServerConnected(false);
      }

      const wfRes = await fetch('/api/workflows');
      if (wfRes.ok) {
        const wfData = await wfRes.json();
        setWorkflows(wfData);
        setSelectedWorkflow((current) => {
          if (wfData.length === 0) return null;
          if (!current) return wfData[0];
          return wfData.find((wf) => wf.name === current.name) || wfData[0];
        });
      }

      const toolRes = await fetch('/api/tools');
      if (toolRes.ok) {
        const toolData = await toolRes.json();
        setAvailableTools(toolData);
      }
    } catch (err) {
      setServerConnected(false);
    }
  };

  useEffect(() => {
    fetchInitialData();
    const interval = setInterval(fetchInitialData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        serverConnected={serverConnected}
      />

      <main className="main-content">
        {activeTab === 'orchestrator' && (
          <>
            <DagGraph
              workflow={selectedWorkflow}
              nodeStatuses={nodeStatuses}
              activeNodeId={activeNodeId}
            />

            <ExecutionConsole
              workflows={workflows}
              selectedWorkflow={selectedWorkflow}
              setSelectedWorkflow={setSelectedWorkflow}
              onRunStart={(status) => setRunStatus(status)}
              runStatus={runStatus}
              logs={logs}
              setLogs={setLogs}
              nodeStatuses={nodeStatuses}
              setNodeStatuses={setNodeStatuses}
              activeNodeId={activeNodeId}
              setActiveNodeId={setActiveNodeId}
              finalOutput={finalOutput}
              setFinalOutput={setFinalOutput}
            />
          </>
        )}

        {activeTab === 'builder' && (
          <WorkflowEditor
            workflows={workflows}
            availableTools={availableTools}
            onWorkflowSaved={fetchInitialData}
          />
        )}

        {activeTab === 'artifacts' && <ArtifactViewer />}
      </main>
    </div>
  );
}
