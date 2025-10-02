'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Task, Agent, TaskStatus, TaskPriority } from '@/types';
import { api } from '@/lib/uiApi';

interface TaskManagementPanelProps {
  className?: string;
}

export const TaskManagementPanel: React.FC<TaskManagementPanelProps> = ({ className = '' }) => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | TaskStatus>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCreatingTask, setIsCreatingTask] = useState(false);
  const [newTask, setNewTask] = useState({
    title: '',
    description: '',
    priority: 'medium' as TaskPriority,
    assignedAgent: '',
    dueDate: '',
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setError(null);
      const [taskList, agentList] = await Promise.all([
        api.getTasks(),
        api.getAgents(),
      ]);
      setTasks(taskList);
      setAgents(agentList);
    } catch (err) {
      setError('Failed to load data');
      console.error('Error loading data:', err);
    } finally {
      setLoading(false);
    }
  };

  const createTask = async () => {
    if (!newTask.title.trim()) return;

    try {
      const task = await api.createTask({
        title: newTask.title,
        description: newTask.description,
        priority: newTask.priority,
        assignedAgent: newTask.assignedAgent || undefined,
        dueDate: newTask.dueDate ? new Date(newTask.dueDate) : undefined,
      });
      setTasks(prev => [...prev, task]);
      setIsCreatingTask(false);
      setNewTask({
        title: '',
        description: '',
        priority: 'medium',
        assignedAgent: '',
        dueDate: '',
      });
    } catch (err) {
      setError('Failed to create task');
    }
  };

  const updateTaskStatus = async (taskId: string, status: TaskStatus) => {
    try {
      const updatedTask = await api.updateTaskStatus(taskId, status);
      setTasks(prev => prev.map(t => t.id === taskId ? updatedTask : t));
    } catch (err) {
      setError('Failed to update task status');
    }
  };

  const deleteTask = async (taskId: string) => {
    if (!confirm('Are you sure you want to delete this task?')) return;

    try {
      await api.deleteTask(taskId);
      setTasks(prev => prev.filter(t => t.id !== taskId));
      if (selectedTask === taskId) {
        setSelectedTask(null);
      }
    } catch (err) {
      setError('Failed to delete task');
    }
  };

  const assignTask = async (taskId: string, agentId: string) => {
    try {
      const updatedTask = await api.assignTask(taskId, agentId);
      setTasks(prev => prev.map(t => t.id === taskId ? updatedTask : t));
    } catch (err) {
      setError('Failed to assign task');
    }
  };

  const getStatusColor = (status: TaskStatus) => {
    switch (status) {
      case 'completed':
        return 'text-green-700 bg-green-100 dark:bg-green-900 dark:text-green-200';
      case 'in-progress':
        return 'text-blue-700 bg-blue-100 dark:bg-blue-900 dark:text-blue-200';
      case 'pending':
        return 'text-yellow-700 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-200';
      case 'blocked':
        return 'text-red-700 bg-red-100 dark:bg-red-900 dark:text-red-200';
      default:
        return 'text-gray-700 bg-gray-100 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getPriorityColor = (priority: TaskPriority) => {
    switch (priority) {
      case 'high':
        return 'text-red-600';
      case 'medium':
        return 'text-yellow-600';
      case 'low':
        return 'text-green-600';
      default:
        return 'text-gray-600';
    }
  };

  const getPriorityIcon = (priority: TaskPriority) => {
    switch (priority) {
      case 'high':
        return '🔴';
      case 'medium':
        return '🟡';
      case 'low':
        return '🟢';
      default:
        return '⚪';
    }
  };

  const filteredTasks = filter === 'all' ? tasks : tasks.filter(task => task.status === filter);
  const selectedTaskData = selectedTask ? tasks.find(t => t.id === selectedTask) : null;

  if (loading) {
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded mb-4"></div>
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-300 dark:bg-gray-700 rounded mb-2"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <div className="text-red-500 mb-2">⚠️ {error}</div>
        <button onClick={loadData} className="btn btn-primary">Retry</button>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Task Management</h2>
        <div className="flex items-center space-x-2">
          <select
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-sm"
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
          >
            <option value="all">All Tasks</option>
            <option value="pending">Pending</option>
            <option value="in-progress">In Progress</option>
            <option value="completed">Completed</option>
            <option value="blocked">Blocked</option>
          </select>
          <button
            onClick={() => setIsCreatingTask(true)}
            className="btn btn-primary"
          >
            + New Task
          </button>
        </div>
      </div>

      {/* Task Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {(['pending', 'in-progress', 'completed', 'blocked'] as TaskStatus[]).map((status) => {
          const count = tasks.filter(t => t.status === status).length;
          return (
            <div 
              key={status} 
              className={`card p-4 cursor-pointer transition-all hover:shadow-md ${
                filter === status ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setFilter(status)}
            >
              <div className="text-center">
                <div className="text-2xl font-bold">{count}</div>
                <div className={`text-sm font-medium capitalize ${getStatusColor(status).split(' ')[0]}`}>
                  {status.replace('-', ' ')}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Task List */}
        <div className="lg:col-span-2 space-y-2">
          {filteredTasks.length === 0 ? (
            <div className="card p-8 text-center text-gray-500">
              <div className="text-4xl mb-2">📋</div>
              <p>No tasks found for the selected filter</p>
            </div>
          ) : (
            filteredTasks.map((task) => {
              const assignedAgent = task.assignedAgent ? agents.find(a => a.id === task.assignedAgent) : null;
              
              return (
                <motion.div
                  key={task.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`card p-4 cursor-pointer transition-all hover:shadow-lg ${
                    selectedTask === task.id ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => setSelectedTask(task.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="text-lg">{getPriorityIcon(task.priority)}</span>
                        <h3 className="font-medium truncate">{task.title}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(task.status)}`}>
                          {task.status.replace('-', ' ')}
                        </span>
                      </div>
                      
                      {task.description && (
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 line-clamp-2">
                          {task.description}
                        </p>
                      )}
                      
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        {assignedAgent && (
                          <div className="flex items-center space-x-1">
                            <span>{assignedAgent.avatar}</span>
                            <span>{assignedAgent.name}</span>
                          </div>
                        )}
                        {task.dueDate && (
                          <div>Due: {new Date(task.dueDate).toLocaleDateString()}</div>
                        )}
                        <div>Created: {new Date(task.createdAt).toLocaleDateString()}</div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2 ml-4">
                      <select
                        className="text-xs px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        value={task.status}
                        onChange={(e) => {
                          e.stopPropagation();
                          updateTaskStatus(task.id, e.target.value as TaskStatus);
                        }}
                      >
                        <option value="pending">Pending</option>
                        <option value="in-progress">In Progress</option>
                        <option value="completed">Completed</option>
                        <option value="blocked">Blocked</option>
                      </select>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteTask(task.id);
                        }}
                        className="text-gray-400 hover:text-red-500 text-xs p-1"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                </motion.div>
              );
            })
          )}
        </div>

        {/* Task Details Sidebar */}
        <div className="space-y-4">
          {selectedTaskData ? (
            <motion.div
              key={selectedTaskData.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="card p-4"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">{getPriorityIcon(selectedTaskData.priority)}</span>
                  <div>
                    <h3 className="font-semibold">{selectedTaskData.title}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedTaskData.status)}`}>
                      {selectedTaskData.status.replace('-', ' ')}
                    </span>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-sm mb-2">Description</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedTaskData.description || 'No description provided'}
                  </p>
                </div>

                <div>
                  <h4 className="font-medium text-sm mb-2">Assigned Agent</h4>
                  {selectedTaskData.assignedAgent ? (
                    <div className="flex items-center space-x-2">
                      {(() => {
                        const agent = agents.find(a => a.id === selectedTaskData.assignedAgent);
                        return agent ? (
                          <>
                            <span className="text-xl">{agent.avatar}</span>
                            <span className="text-sm">{agent.name}</span>
                          </>
                        ) : (
                          <span className="text-sm text-gray-500">Agent not found</span>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <p className="text-sm text-gray-500 mb-2">No agent assigned</p>
                      <select
                        className="w-full text-xs px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        onChange={(e) => {
                          if (e.target.value) {
                            assignTask(selectedTaskData.id, e.target.value);
                          }
                        }}
                      >
                        <option value="">Select agent...</option>
                        {agents.filter(a => a.status === 'active' || a.status === 'idle').map(agent => (
                          <option key={agent.id} value={agent.id}>
                            {agent.avatar} {agent.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>

                <div>
                  <h4 className="font-medium text-sm mb-2">Priority</h4>
                  <div className={`text-sm ${getPriorityColor(selectedTaskData.priority)}`}>
                    {getPriorityIcon(selectedTaskData.priority)} {selectedTaskData.priority.toUpperCase()}
                  </div>
                </div>

                {selectedTaskData.dueDate && (
                  <div>
                    <h4 className="font-medium text-sm mb-2">Due Date</h4>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {new Date(selectedTaskData.dueDate).toLocaleDateString()}
                    </div>
                  </div>
                )}

                <div>
                  <h4 className="font-medium text-sm mb-2">Timeline</h4>
                  <div className="space-y-1 text-xs text-gray-500">
                    <div>Created: {new Date(selectedTaskData.createdAt).toLocaleString()}</div>
                    <div>Updated: {new Date(selectedTaskData.updatedAt).toLocaleString()}</div>
                    {selectedTaskData.completedAt && (
                      <div>Completed: {new Date(selectedTaskData.completedAt).toLocaleString()}</div>
                    )}
                  </div>
                </div>

                {selectedTaskData.tags && selectedTaskData.tags.length > 0 && (
                  <div>
                    <h4 className="font-medium text-sm mb-2">Tags</h4>
                    <div className="flex flex-wrap gap-1">
                      {selectedTaskData.tags.map((tag, index) => (
                        <span 
                          key={index}
                          className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs rounded"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-medium text-sm mb-2">Quick Actions</h4>
                <div className="space-y-1">
                  <button className="w-full text-left text-xs px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                    ✏️ Edit Task
                  </button>
                  <button className="w-full text-left text-xs px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                    👥 Reassign
                  </button>
                  <button className="w-full text-left text-xs px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                    💬 Add Comment
                  </button>
                  <button 
                    className="w-full text-left text-xs px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded text-red-600"
                    onClick={() => deleteTask(selectedTaskData.id)}
                  >
                    🗑️ Delete Task
                  </button>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="card p-8 text-center text-gray-500">
              <div className="text-4xl mb-2">👆</div>
              <p>Select a task to view details</p>
            </div>
          )}

          {/* Quick Task Creation */}
          <div className="card p-4">
            <h3 className="font-medium mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <button 
                onClick={() => setIsCreatingTask(true)}
                className="btn btn-primary w-full text-sm"
              >
                + Create Task
              </button>
              <button className="btn btn-secondary w-full text-sm">
                📊 View Analytics
              </button>
              <button className="btn btn-secondary w-full text-sm">
                📋 Export Tasks
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Create Task Modal */}
      <AnimatePresence>
        {isCreatingTask && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setIsCreatingTask(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-semibold mb-4">Create New Task</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Task Title *
                  </label>
                  <input
                    type="text"
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                    value={newTask.title}
                    onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                    placeholder="Enter task title"
                    autoFocus
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Description
                  </label>
                  <textarea
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                    rows={3}
                    value={newTask.description}
                    onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                    placeholder="Describe the task..."
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Priority
                    </label>
                    <select
                      className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                      value={newTask.priority}
                      onChange={(e) => setNewTask({ ...newTask, priority: e.target.value as TaskPriority })}
                    >
                      <option value="low">🟢 Low</option>
                      <option value="medium">🟡 Medium</option>
                      <option value="high">🔴 High</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Assign To
                    </label>
                    <select
                      className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                      value={newTask.assignedAgent}
                      onChange={(e) => setNewTask({ ...newTask, assignedAgent: e.target.value })}
                    >
                      <option value="">Unassigned</option>
                      {agents.filter(a => a.status === 'active' || a.status === 'idle').map(agent => (
                        <option key={agent.id} value={agent.id}>
                          {agent.avatar} {agent.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Due Date
                  </label>
                  <input
                    type="date"
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                    value={newTask.dueDate}
                    onChange={(e) => setNewTask({ ...newTask, dueDate: e.target.value })}
                  />
                </div>

                <div className="flex space-x-3">
                  <button 
                    onClick={createTask}
                    className="btn btn-primary flex-1"
                    disabled={!newTask.title.trim()}
                  >
                    Create Task
                  </button>
                  <button 
                    onClick={() => setIsCreatingTask(false)}
                    className="btn btn-secondary flex-1"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};