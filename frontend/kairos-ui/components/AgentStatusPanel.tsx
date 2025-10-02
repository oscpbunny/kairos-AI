'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Agent, AgentStatus } from '@/types';
import { api } from '@/lib/uiApi';

interface AgentStatusPanelProps {
  className?: string;
}

export const AgentStatusPanel: React.FC<AgentStatusPanelProps> = ({ className = '' }) => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAgents();
    const interval = setInterval(loadAgents, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadAgents = async () => {
    try {
      setError(null);
      const agentList = await api.getAgents();
      setAgents(agentList);
    } catch (err) {
      setError('Failed to load agents');
      console.error('Error loading agents:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: AgentStatus) => {
    switch (status) {
      case 'active':
        return 'text-green-500 bg-green-100 dark:bg-green-900';
      case 'idle':
        return 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900';
      case 'busy':
        return 'text-blue-500 bg-blue-100 dark:bg-blue-900';
      case 'offline':
        return 'text-red-500 bg-red-100 dark:bg-red-900';
      default:
        return 'text-gray-500 bg-gray-100 dark:bg-gray-900';
    }
  };

  const getStatusIcon = (status: AgentStatus) => {
    switch (status) {
      case 'active':
        return '🟢';
      case 'idle':
        return '🟡';
      case 'busy':
        return '🔵';
      case 'offline':
        return '🔴';
      default:
        return '⚫';
    }
  };

  const toggleAgentStatus = async (agentId: string) => {
    try {
      const agent = agents.find(a => a.id === agentId);
      if (!agent) return;

      const newStatus: AgentStatus = agent.status === 'active' ? 'idle' : 'active';
      await api.updateAgentStatus(agentId, newStatus);
      
      setAgents(prev => prev.map(a => 
        a.id === agentId ? { ...a, status: newStatus } : a
      ));
    } catch (err) {
      setError('Failed to update agent status');
    }
  };

  if (loading) {
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded mb-4"></div>
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-20 bg-gray-300 dark:bg-gray-700 rounded mb-2"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <div className="text-red-500 mb-2">⚠️ {error}</div>
        <button 
          onClick={loadAgents}
          className="btn btn-primary"
        >
          Retry
        </button>
      </div>
    );
  }

  const selectedAgentData = selectedAgent ? agents.find(a => a.id === selectedAgent) : null;

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Agent Status Monitor</h2>
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          <span>Auto-refresh: 5s</span>
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Agent List */}
        <div className="lg:col-span-2 space-y-2">
          {agents.map((agent) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`card p-4 cursor-pointer transition-all hover:shadow-lg ${
                selectedAgent === agent.id ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setSelectedAgent(agent.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="text-2xl">{agent.avatar}</div>
                  <div>
                    <div className="font-medium">{agent.name}</div>
                    <div className="text-sm text-gray-500">{agent.type}</div>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <span 
                    className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}
                  >
                    {getStatusIcon(agent.status)} {agent.status.toUpperCase()}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleAgentStatus(agent.id);
                    }}
                    className="text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 rounded transition-colors"
                  >
                    Toggle
                  </button>
                </div>
              </div>
              
              {agent.currentTask && (
                <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  <strong>Current Task:</strong> {agent.currentTask}
                </div>
              )}
              
              <div className="mt-2 flex flex-wrap gap-1">
                {agent.capabilities.slice(0, 3).map((capability, index) => (
                  <span 
                    key={index}
                    className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded"
                  >
                    {capability}
                  </span>
                ))}
                {agent.capabilities.length > 3 && (
                  <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded">
                    +{agent.capabilities.length - 3} more
                  </span>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Agent Details */}
        <div className="space-y-4">
          {selectedAgentData ? (
            <motion.div
              key={selectedAgentData.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="card p-4"
            >
              <div className="text-center mb-4">
                <div className="text-4xl mb-2">{selectedAgentData.avatar}</div>
                <h3 className="font-semibold text-lg">{selectedAgentData.name}</h3>
                <p className="text-gray-500 text-sm">{selectedAgentData.type}</p>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Status</h4>
                  <span 
                    className={`px-3 py-2 rounded-lg text-sm font-medium ${getStatusColor(selectedAgentData.status)}`}
                  >
                    {getStatusIcon(selectedAgentData.status)} {selectedAgentData.status.toUpperCase()}
                  </span>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Description</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedAgentData.description}
                  </p>
                </div>

                {selectedAgentData.currentTask && (
                  <div>
                    <h4 className="font-medium mb-2">Current Task</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {selectedAgentData.currentTask}
                    </p>
                  </div>
                )}

                <div>
                  <h4 className="font-medium mb-2">Capabilities</h4>
                  <div className="space-y-1">
                    {selectedAgentData.capabilities.map((capability, index) => (
                      <div 
                        key={index}
                        className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded"
                      >
                        {capability}
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Performance Metrics</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Tasks Completed:</span>
                      <span className="font-medium">
                        {Math.floor(Math.random() * 100) + 50}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Success Rate:</span>
                      <span className="font-medium text-green-600">
                        {(Math.random() * 15 + 85).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Avg Response Time:</span>
                      <span className="font-medium">
                        {(Math.random() * 2 + 0.5).toFixed(1)}s
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="card p-8 text-center text-gray-500">
              <div className="text-4xl mb-2">👥</div>
              <p>Select an agent to view detailed information</p>
            </div>
          )}

          {/* Quick Actions */}
          <div className="card p-4">
            <h3 className="font-medium mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <button className="btn btn-primary w-full text-sm">
                Create New Agent
              </button>
              <button className="btn btn-secondary w-full text-sm">
                Agent Health Check
              </button>
              <button className="btn btn-secondary w-full text-sm">
                Export Agent Logs
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Status Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        {(['active', 'idle', 'busy', 'offline'] as AgentStatus[]).map((status) => {
          const count = agents.filter(a => a.status === status).length;
          return (
            <div key={status} className="card p-4 text-center">
              <div className="text-2xl mb-1">{getStatusIcon(status)}</div>
              <div className="text-2xl font-bold">{count}</div>
              <div className={`text-sm font-medium ${getStatusColor(status).split(' ')[0]}`}>
                {status.toUpperCase()}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};