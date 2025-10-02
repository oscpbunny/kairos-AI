'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Project, ProjectContext } from '@/types';
import { api } from '@/lib/uiApi';

interface ProjectContextPanelProps {
  className?: string;
}

export const ProjectContextPanel: React.FC<ProjectContextPanelProps> = ({ className = '' }) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'context' | 'goals' | 'config'>('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setError(null);
      const projectList = await api.getProjects();
      setProjects(projectList);
      if (!selectedProject && projectList.length > 0) {
        setSelectedProject(projectList[0].id);
      }
    } catch (err) {
      setError('Failed to load projects');
      console.error('Error loading projects:', err);
    } finally {
      setLoading(false);
    }
  };

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    
    try {
      const newProject = await api.createProject({
        name: newProjectName,
        description: newProjectDescription,
      });
      setProjects(prev => [...prev, newProject]);
      setSelectedProject(newProject.id);
      setIsCreatingProject(false);
      setNewProjectName('');
      setNewProjectDescription('');
    } catch (err) {
      setError('Failed to create project');
    }
  };

  const deleteProject = async (projectId: string) => {
    if (!confirm('Are you sure you want to delete this project?')) return;
    
    try {
      await api.deleteProject(projectId);
      setProjects(prev => prev.filter(p => p.id !== projectId));
      if (selectedProject === projectId) {
        setSelectedProject(projects.find(p => p.id !== projectId)?.id || null);
      }
    } catch (err) {
      setError('Failed to delete project');
    }
  };

  const updateProjectContext = async (projectId: string, context: Partial<ProjectContext>) => {
    try {
      const updatedProject = await api.updateProjectContext(projectId, context);
      setProjects(prev => prev.map(p => p.id === projectId ? updatedProject : p));
    } catch (err) {
      setError('Failed to update project context');
    }
  };

  const selectedProjectData = selectedProject ? projects.find(p => p.id === selectedProject) : null;

  if (loading) {
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded mb-4"></div>
          <div className="h-32 bg-gray-300 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <div className="text-red-500 mb-2">⚠️ {error}</div>
        <button onClick={loadProjects} className="btn btn-primary">Retry</button>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Project Context Manager</h2>
        <button
          onClick={() => setIsCreatingProject(true)}
          className="btn btn-primary"
        >
          + New Project
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Project Sidebar */}
        <div className="space-y-2">
          <h3 className="font-medium text-sm text-gray-500 uppercase tracking-wide">Projects</h3>
          {projects.map((project) => (
            <motion.div
              key={project.id}
              whileHover={{ scale: 1.02 }}
              className={`card p-3 cursor-pointer transition-all ${
                selectedProject === project.id 
                  ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                  : 'hover:shadow-md'
              }`}
              onClick={() => setSelectedProject(project.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate">{project.name}</div>
                  <div className="text-xs text-gray-500 mt-1 line-clamp-2">
                    {project.description || 'No description'}
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteProject(project.id);
                  }}
                  className="text-gray-400 hover:text-red-500 text-xs ml-2"
                >
                  ×
                </button>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                <span>{project.context?.goals?.length || 0} goals</span>
                <span>{project.status}</span>
              </div>
            </motion.div>
          ))}

          {projects.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <div className="text-2xl mb-2">📁</div>
              <p className="text-sm">No projects yet</p>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-4">
          {selectedProjectData ? (
            <>
              {/* Project Header */}
              <div className="card p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-semibold">{selectedProjectData.name}</h3>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">
                      {selectedProjectData.description || 'No description provided'}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      selectedProjectData.status === 'active' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : selectedProjectData.status === 'paused'
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                    }`}>
                      {selectedProjectData.status}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded">
                    <div className="text-2xl font-bold text-blue-600">
                      {selectedProjectData.context?.goals?.length || 0}
                    </div>
                    <div className="text-sm text-gray-600">Goals</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded">
                    <div className="text-2xl font-bold text-green-600">
                      {selectedProjectData.context?.completedTasks || 0}
                    </div>
                    <div className="text-sm text-gray-600">Completed</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded">
                    <div className="text-2xl font-bold text-orange-600">
                      {selectedProjectData.context?.activeTasks || 0}
                    </div>
                    <div className="text-sm text-gray-600">Active</div>
                  </div>
                </div>
              </div>

              {/* Tab Navigation */}
              <div className="flex space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
                {[
                  { key: 'overview', label: '📊 Overview', icon: '📊' },
                  { key: 'context', label: '📋 Context', icon: '📋' },
                  { key: 'goals', label: '🎯 Goals', icon: '🎯' },
                  { key: 'config', label: '⚙️ Config', icon: '⚙️' }
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key as any)}
                    className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all ${
                      activeTab === tab.key
                        ? 'bg-white dark:bg-gray-700 text-blue-600 shadow-sm'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                >
                  {activeTab === 'overview' && (
                    <div className="space-y-4">
                      <div className="card p-4">
                        <h4 className="font-medium mb-3">Project Timeline</h4>
                        <div className="space-y-2">
                          <div className="flex items-center space-x-3 text-sm">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="text-gray-500">Created:</span>
                            <span>{new Date(selectedProjectData.createdAt).toLocaleDateString()}</span>
                          </div>
                          <div className="flex items-center space-x-3 text-sm">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            <span className="text-gray-500">Updated:</span>
                            <span>{new Date(selectedProjectData.updatedAt).toLocaleDateString()}</span>
                          </div>
                        </div>
                      </div>

                      <div className="card p-4">
                        <h4 className="font-medium mb-3">Recent Activity</h4>
                        <div className="space-y-2">
                          {[
                            { action: 'Goal completed', time: '2 hours ago', type: 'success' },
                            { action: 'New task assigned', time: '4 hours ago', type: 'info' },
                            { action: 'Context updated', time: '1 day ago', type: 'warning' }
                          ].map((activity, i) => (
                            <div key={i} className="flex items-center space-x-3 text-sm">
                              <div className={`w-2 h-2 rounded-full ${
                                activity.type === 'success' ? 'bg-green-500' :
                                activity.type === 'info' ? 'bg-blue-500' : 'bg-yellow-500'
                              }`}></div>
                              <span>{activity.action}</span>
                              <span className="text-gray-500">•</span>
                              <span className="text-gray-500">{activity.time}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'context' && (
                    <div className="card p-6">
                      <h4 className="font-medium mb-4">Project Context</h4>
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Background Information
                          </label>
                          <textarea
                            className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                            rows={4}
                            defaultValue={selectedProjectData.description || ''}
                            placeholder="Describe the project background, requirements, and context..."
                            onBlur={(e) => updateProjectContext(selectedProjectData.id, { description: e.target.value })}
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Key Stakeholders
                          </label>
                          <input
                            className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                            defaultValue={''}
                            placeholder="Enter stakeholder names (comma separated)"
                            onBlur={(e) => console.log('Stakeholders:', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Success Criteria
                          </label>
                          <textarea
                            className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                            rows={3}
                            defaultValue={''}
                            placeholder="Define what success looks like for this project..."
                            onBlur={(e) => console.log('Success criteria:', e.target.value)}
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'goals' && (
                    <div className="card p-6">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="font-medium">Project Goals</h4>
                        <button className="btn btn-primary btn-sm">+ Add Goal</button>
                      </div>
                      <div className="space-y-3">
                        {selectedProjectData.context?.goals?.map((goal, index) => (
                          <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-800 rounded">
                            <input type="checkbox" className="rounded" />
                            <div className="flex-1">
                              <div className="font-medium">{goal}</div>
                            </div>
                          </div>
                        )) || (
                          <div className="text-center py-8 text-gray-500">
                            <div className="text-2xl mb-2">🎯</div>
                            <p>No goals defined yet</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {activeTab === 'config' && (
                    <div className="card p-6">
                      <h4 className="font-medium mb-4">Project Configuration</h4>
                      <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                              Project Type
                            </label>
                            <select className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800">
                              <option>Software Development</option>
                              <option>Research</option>
                              <option>Marketing</option>
                              <option>Other</option>
                            </select>
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                              Priority Level
                            </label>
                            <select className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800">
                              <option>High</option>
                              <option>Medium</option>
                              <option>Low</option>
                            </select>
                          </div>
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Auto-assign Agents
                          </label>
                          <div className="space-y-2">
                            {['Development Agent', 'Research Agent', 'QA Agent'].map((agent) => (
                              <label key={agent} className="flex items-center space-x-2">
                                <input type="checkbox" className="rounded" />
                                <span className="text-sm">{agent}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            </>
          ) : (
            <div className="card p-8 text-center text-gray-500">
              <div className="text-4xl mb-2">📋</div>
              <p>Select a project to view its context and configuration</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Project Modal */}
      <AnimatePresence>
        {isCreatingProject && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setIsCreatingProject(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-semibold mb-4">Create New Project</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Project Name
                  </label>
                  <input
                    type="text"
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    placeholder="Enter project name"
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
                    value={newProjectDescription}
                    onChange={(e) => setNewProjectDescription(e.target.value)}
                    placeholder="Describe your project..."
                  />
                </div>
                <div className="flex space-x-3">
                  <button 
                    onClick={createProject}
                    className="btn btn-primary flex-1"
                    disabled={!newProjectName.trim()}
                  >
                    Create Project
                  </button>
                  <button 
                    onClick={() => setIsCreatingProject(false)}
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