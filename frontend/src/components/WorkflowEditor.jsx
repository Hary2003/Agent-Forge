import React, { useEffect, useState } from 'react';
import { Plus, Trash2, Save, FileCode, CheckCircle2, AlertCircle } from 'lucide-react';
import yaml from 'js-yaml';

export default function WorkflowEditor({ workflows, availableTools, onWorkflowSaved }) {
  const [selectedName, setSelectedName] = useState(workflows[0]?.name || 'new_workflow');
  const [editingWorkflow, setEditingWorkflow] = useState(
    workflows[0] || {
      name: 'new_workflow',
      description: 'A new multi-agent DAG workflow',
      nodes: [
        {
          id: 'researcher',
          role: 'researcher',
          goal: 'Investigate the task topic and outline main points.',
          inputs: [],
          tools: [],
          max_iterations: 4,
        },
      ],
    }
  );
  const [rawYaml, setRawYaml] = useState('');
  const [viewMode, setViewMode] = useState('visual'); // 'visual' | 'yaml'
  const [saveStatus, setSaveStatus] = useState(null);

  useEffect(() => {
    if (workflows.length === 0) return;
    const selectedStillExists = workflows.some((workflow) => workflow.name === selectedName);
    if (selectedStillExists) return;

    setSelectedName(workflows[0].name);
    setEditingWorkflow(workflows[0]);
    setRawYaml(yaml.dump(workflows[0]));
  }, [workflows, selectedName]);

  const handleSelectWorkflow = (wfName) => {
    setSelectedName(wfName);
    const found = workflows.find((w) => w.name === wfName);
    if (found) {
      setEditingWorkflow(found);
      setRawYaml(yaml.dump(found));
    }
  };

  const handleAddNode = () => {
    const newNodeId = `node_${editingWorkflow.nodes.length + 1}`;
    setEditingWorkflow((prev) => ({
      ...prev,
      nodes: [
        ...prev.nodes,
        {
          id: newNodeId,
          role: 'analyst',
          goal: 'Process context and summarize key metrics.',
          inputs: prev.nodes.length > 0 ? [prev.nodes[prev.nodes.length - 1].id] : [],
          tools: [],
          max_iterations: 4,
        },
      ],
    }));
  };

  const handleUpdateNode = (index, field, value) => {
    setEditingWorkflow((prev) => {
      const updatedNodes = [...prev.nodes];
      updatedNodes[index] = { ...updatedNodes[index], [field]: value };
      return { ...prev, nodes: updatedNodes };
    });
  };

  const handleDeleteNode = (index) => {
    setEditingWorkflow((prev) => ({
      ...prev,
      nodes: prev.nodes.filter((_, i) => i !== index),
    }));
  };

  const handleSave = async () => {
    let payload = editingWorkflow;
    if (viewMode === 'yaml') {
      try {
        payload = yaml.load(rawYaml);
      } catch (err) {
        setSaveStatus({ type: 'error', msg: `YAML Syntax Error: ${err.message}` });
        return;
      }
    }

    try {
      const res = await fetch('/api/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('Failed to save workflow to backend');
      setSaveStatus({ type: 'success', msg: `Workflow '${payload.name}' saved successfully!` });
      onWorkflowSaved();
      setTimeout(() => setSaveStatus(null), 4000);
    } catch (err) {
      setSaveStatus({ type: 'error', msg: err.message });
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Action Bar */}
      <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="input-group">
            <select
              className="select-input"
              value={selectedName}
              onChange={(e) => handleSelectWorkflow(e.target.value)}
            >
              {workflows.map((w) => (
                <option key={w.name} value={w.name}>
                  📂 {w.name}.yaml
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              className={`btn ${viewMode === 'visual' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('visual')}
            >
              Visual Builder
            </button>
            <button
              className={`btn ${viewMode === 'yaml' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => {
                setRawYaml(yaml.dump(editingWorkflow));
                setViewMode('yaml');
              }}
            >
              <FileCode size={16} /> YAML View
            </button>
          </div>
        </div>

        <button className="btn btn-primary" onClick={handleSave}>
          <Save size={16} /> Save Workflow
        </button>
      </div>

      {saveStatus && (
        <div
          style={{
            padding: '0.9rem 1.25rem',
            borderRadius: '8px',
            background: saveStatus.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
            border: `1px solid ${saveStatus.type === 'success' ? '#10b981' : '#ef4444'}`,
            color: saveStatus.type === 'success' ? '#10b981' : '#ef4444',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          {saveStatus.type === 'success' ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
          {saveStatus.msg}
        </div>
      )}

      {/* Editor Content Body */}
      {viewMode === 'visual' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="input-group">
              <label className="input-label">Workflow Name</label>
              <input
                type="text"
                className="text-input"
                value={editingWorkflow.name}
                onChange={(e) => setEditingWorkflow({ ...editingWorkflow, name: e.target.value })}
              />
            </div>
            <div className="input-group">
              <label className="input-label">Description</label>
              <input
                type="text"
                className="text-input"
                value={editingWorkflow.description}
                onChange={(e) => setEditingWorkflow({ ...editingWorkflow, description: e.target.value })}
              />
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
              DAG Agent Nodes ({editingWorkflow.nodes.length})
            </div>
            <button className="btn btn-secondary" onClick={handleAddNode}>
              <Plus size={16} /> Add Node
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '1.5rem' }}>
            {editingWorkflow.nodes.map((node, index) => (
              <div key={index} className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>
                    Node #{index + 1}: {node.id}
                  </div>
                  <button
                    className="btn btn-secondary"
                    style={{ padding: '0.3rem 0.5rem', color: '#ef4444' }}
                    onClick={() => handleDeleteNode(index)}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>

                <div className="input-group">
                  <label className="input-label">Node ID</label>
                  <input
                    type="text"
                    className="text-input"
                    value={node.id}
                    onChange={(e) => handleUpdateNode(index, 'id', e.target.value)}
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Role</label>
                  <input
                    type="text"
                    className="text-input"
                    value={node.role}
                    onChange={(e) => handleUpdateNode(index, 'role', e.target.value)}
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Goal / Prompt Instructions</label>
                  <textarea
                    className="textarea-input"
                    rows={3}
                    value={node.goal}
                    onChange={(e) => handleUpdateNode(index, 'goal', e.target.value)}
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Upstream Dependencies (Inputs)</label>
                  <input
                    type="text"
                    className="text-input"
                    placeholder="Comma separated node IDs (e.g. researcher, analyst)"
                    value={Array.isArray(node.inputs) ? node.inputs.join(', ') : ''}
                    onChange={(e) =>
                      handleUpdateNode(
                        index,
                        'inputs',
                        e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                      )
                    }
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Allowed Tools</label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.2rem' }}>
                    {availableTools.map((tool) => {
                      const isSelected = (node.tools || []).includes(tool.name);
                      return (
                        <button
                          key={tool.name}
                          type="button"
                          className={`btn ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                          style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                          onClick={() => {
                            const current = node.tools || [];
                            const updated = isSelected
                              ? current.filter((t) => t !== tool.name)
                              : [...current, tool.name];
                            handleUpdateNode(index, 'tools', updated);
                          }}
                        >
                          {tool.name}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <textarea
            className="textarea-input"
            rows={20}
            style={{ width: '100%', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}
            value={rawYaml}
            onChange={(e) => setRawYaml(e.target.value)}
          />
        </div>
      )}
    </div>
  );
}
