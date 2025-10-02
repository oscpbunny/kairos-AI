'use client';

import { useState, useEffect } from 'react';
import { 
  MessageCircle, 
  Upload, 
  Settings, 
  Users, 
  BarChart3, 
  FileText, 
  Folder, 
  Brain, 
  Zap,
  Play,
  Pause,
  RotateCcw,
  Monitor,
  Database,
  Globe,
  Terminal,
  Code,
  Lightbulb,
  Target,
  CheckCircle,
  AlertTriangle,
  Activity,
  Clock,
  TrendingUp,
  GitBranch,
  Layers,
  Cloud,
  Shield,
  Cpu,
  HardDrive,
  Network,
  Eye,
  BookOpen,
  Wrench,
  Search,
  Filter,
  Download,
  Share,
  Plus,
  Minus,
  X,
  ChevronRight,
  ChevronDown,
  ExternalLink
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { RealTimeChatInterface } from '@/components/RealTimeChatInterface';
import { FileUploadZone } from '@/components/FileUploadZone';
import { AgentStatusPanel } from '@/components/AgentStatusPanel';
import { ProjectContextPanel } from '@/components/ProjectContextPanel';
import { SystemMetricsPanel } from '@/components/SystemMetricsPanel';
import { TaskManagementPanel } from '@/components/TaskManagementPanel';
import { AnalyticsPanel } from '@/components/AnalyticsPanel';
import { SettingsPanel } from '@/components/SettingsPanel';
import { DatabasePanel } from '@/components/DatabasePanel';
import { LogsPanel } from '@/components/LogsPanel';
import { DockerPanel } from '@/components/DockerPanel';
import { APIPanel } from '@/components/APIPanel';
import { ConsciousnessVisualization } from '@/components/ConsciousnessVisualization';
import { mockAgents, createMockProject } from '@/lib/uiApi';

type Tab = 
  | 'chat' 
  | 'agents' 
  | 'project' 
  | 'files' 
  | 'tasks' 
  | 'analytics' 
  | 'consciousness'
  | 'system'
  | 'database'
  | 'logs'
  | 'docker'
  | 'api'
  | 'settings';

interface TabConfig {
  id: Tab;
  name: string;
  icon: React.ReactNode;
  description: string;
  category: 'core' | 'management' | 'development' | 'system';
}

const tabs: TabConfig[] = [
  // Core functionality
  { id: 'chat', name: 'AI Chat', icon: <MessageCircle />, description: 'Multi-agent conversation interface', category: 'core' },
  { id: 'agents', name: 'Agents', icon: <Users />, description: 'Agent status and management', category: 'core' },
  { id: 'project', name: 'Project', icon: <Target />, description: 'Project context and goals', category: 'core' },
  { id: 'files', name: 'Files', icon: <FileText />, description: 'File upload and analysis', category: 'core' },
  
  // Management
  { id: 'tasks', name: 'Tasks', icon: <CheckCircle />, description: 'Collaboration task management', category: 'management' },
  { id: 'analytics', name: 'Analytics', icon: <BarChart3 />, description: 'Performance metrics and insights', category: 'management' },
  { id: 'consciousness', name: 'Consciousness', icon: <Brain />, description: 'AI consciousness visualization', category: 'management' },
  
  // Development
  { id: 'api', name: 'API', icon: <Code />, description: 'REST API documentation and testing', category: 'development' },
  { id: 'database', name: 'Database', icon: <Database />, description: 'Database management and queries', category: 'development' },
  { id: 'logs', name: 'Logs', icon: <Terminal />, description: 'System logs and debugging', category: 'development' },
  
  // System
  { id: 'docker', name: 'Docker', icon: <Layers />, description: 'Container management and deployment', category: 'system' },
  { id: 'system', name: 'System', icon: <Monitor />, description: 'System metrics and health', category: 'system' },
  { id: 'settings', name: 'Settings', icon: <Settings />, description: 'Application configuration', category: 'system' },
];

export default function KairosMainInterface() {
  const [activeTab, setActiveTab] = useState<Tab>('chat');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [systemStatus, setSystemStatus] = useState({
    isConnected: true,
    isLoading: false,
    agentCount: 5,
    activeAgents: 4,
    systemHealth: 'healthy' as 'healthy' | 'warning' | 'error'
  });
  const [notifications, setNotifications] = useState(0);

  // Mock data
  const mockProject = createMockProject();
  const agents = mockAgents();

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setSystemStatus(prev => ({
        ...prev,
        activeAgents: Math.min(5, Math.max(3, prev.activeAgents + (Math.random() > 0.7 ? (Math.random() > 0.5 ? 1 : -1) : 0)))
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getTabsByCategory = (category: string) => {
    return tabs.filter(tab => tab.category === category);
  };

  const renderTabContent = () => {
    const commonProps = {
      className: "h-full animate-fade-in"
    };

    switch (activeTab) {
      case 'chat':
        return <RealTimeChatInterface {...commonProps} />;
      case 'agents':
        return <AgentStatusPanel {...commonProps} />;
      case 'project':
        return <ProjectContextPanel {...commonProps} />;
      case 'files':
        return <FileUploadZone onFileDrop={() => {}} {...commonProps} />;
      case 'tasks':
        return <TaskManagementPanel {...commonProps} />;
      case 'analytics':
        return <AnalyticsPanel {...commonProps} />;
      case 'consciousness':
        return <ConsciousnessVisualization {...commonProps} />;
      case 'system':
        return <SystemMetricsPanel {...commonProps} />;
      case 'database':
        return <DatabasePanel {...commonProps} />;
      case 'logs':
        return <LogsPanel {...commonProps} />;
      case 'docker':
        return <DockerPanel {...commonProps} />;
      case 'api':
        return <APIPanel {...commonProps} />;
      case 'settings':
        return <SettingsPanel {...commonProps} />;
      default:
        return <div className="p-8 text-center text-gray-500">Select a tab to get started</div>;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <motion.div 
        initial={false}
        animate={{ width: isSidebarCollapsed ? 80 : 280 }}
        className="bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col shadow-lg"
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            {!isSidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <h1 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
                  <Brain className="mr-2 text-kairos-500" size={24} />
                  Kairos
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">Multi-Agent AI Platform</p>
              </motion.div>
            )}
            <button
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
              className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <ChevronRight className={`transition-transform ${isSidebarCollapsed ? '' : 'rotate-180'}`} size={16} />
            </button>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            {!isSidebarCollapsed ? (
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${systemStatus.isConnected ? 'agent-online' : 'agent-offline'}`}></div>
                <span className="text-xs text-gray-600 dark:text-gray-300">
                  {systemStatus.activeAgents}/{systemStatus.agentCount} Agents Active
                </span>
              </div>
            ) : (
              <div className="flex justify-center">
                <div className={`w-3 h-3 rounded-full ${systemStatus.isConnected ? 'agent-online' : 'agent-offline'}`}></div>
              </div>
            )}
          </div>
        </div>

        {/* Navigation Categories */}
        <nav className="flex-1 overflow-y-auto p-4 space-y-6">
          {(['core', 'management', 'development', 'system'] as const).map(category => (
            <div key={category}>
              {!isSidebarCollapsed && (
                <h3 className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-3">
                  {category.replace('_', ' ')}
                </h3>
              )}
              <div className="space-y-1">
                {getTabsByCategory(category).map(tab => (
                  <motion.button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      activeTab === tab.id
                        ? 'bg-kairos-100 text-kairos-700 dark:bg-kairos-900/50 dark:text-kairos-300'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                    whileHover={{ x: 2 }}
                    whileTap={{ scale: 0.98 }}
                    title={isSidebarCollapsed ? tab.name : tab.description}
                  >
                    <div className="flex items-center justify-center w-5 h-5 mr-3 flex-shrink-0">
                      {tab.icon}
                    </div>
                    {!isSidebarCollapsed && (
                      <span className="truncate">{tab.name}</span>
                    )}
                  </motion.button>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* Quick Actions */}
        {!isSidebarCollapsed && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="space-y-2">
              <button className="w-full flex items-center px-3 py-2 text-sm text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-lg">
                <Play size={16} className="mr-2" />
                Quick Demo
              </button>
              <button className="w-full flex items-center px-3 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg">
                <ExternalLink size={16} className="mr-2" />
                Open Dashboard
              </button>
            </div>
          </div>
        )}
      </motion.div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                {tabs.find(tab => tab.id === activeTab)?.name || 'Dashboard'}
              </h2>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {tabs.find(tab => tab.id === activeTab)?.description}
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* System Health Indicator */}
              <div className="flex items-center space-x-2">
                <Activity size={16} className={`${
                  systemStatus.systemHealth === 'healthy' ? 'text-green-500' :
                  systemStatus.systemHealth === 'warning' ? 'text-yellow-500' :
                  'text-red-500'
                }`} />
                <span className="text-sm text-gray-600 dark:text-gray-300 capitalize">
                  {systemStatus.systemHealth}
                </span>
              </div>

              {/* Notifications */}
              {notifications > 0 && (
                <div className="relative">
                  <div className="w-4 h-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                    {notifications}
                  </div>
                </div>
              )}

              {/* Theme Toggle & User Menu can go here */}
              <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center">
                <User size={16} className="text-gray-600 dark:text-gray-300" />
              </div>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {renderTabContent()}
            </motion.div>
          </AnimatePresence>
        </main>

        {/* Status Bar */}
        <div className="bg-gray-100 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-2">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-4">
              <span>Connected to Kairos Backend</span>
              <span>•</span>
              <span>{systemStatus.activeAgents} of {systemStatus.agentCount} agents active</span>
              <span>•</span>
              <span className="flex items-center">
                <Clock size={12} className="mr-1" />
                Last sync: {new Date().toLocaleTimeString()}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span>v2.0.0</span>
              <span>•</span>
              <span>Port 3001</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Fix import for User icon
import { User } from 'lucide-react';