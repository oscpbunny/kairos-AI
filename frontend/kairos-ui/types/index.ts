// Core Kairos Types
export type AgentStatus = 'active' | 'idle' | 'busy' | 'offline';

export interface Agent {
  id: string;
  name: string;
  // Original fields (kept for compatibility)
  role?: 'leader' | 'creative' | 'analytical' | 'collaborator' | 'specialist';
  specializations?: string[];
  consciousnessLevel?: number;
  emotionalState?: string;
  active?: boolean;
  lastActivity?: string;
  // UI fields used by panels
  type?: string;
  status: AgentStatus;
  avatar?: string; // emoji or small icon
  description?: string;
  capabilities: string[];
  currentTask?: string;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'agent' | 'system';
  content: string;
  agentId?: string;
  timestamp: Date;
  attachments?: FileAttachment[];
  context?: ProjectContext;
  metadata?: Record<string, any>;
}

export interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
  path: string;
  uploadedAt: Date;
  processed: boolean;
  analysisResult?: FileAnalysis;
}

export interface FileAnalysis {
  summary: string;
  insights: string[];
  suggestions: string[];
  codeMetrics?: CodeMetrics;
  designMetrics?: DesignMetrics;
}

export interface CodeMetrics {
  language: string;
  lines: number;
  complexity: number;
  quality: number;
  issues: Issue[];
}

export interface DesignMetrics {
  fileType: string;
  dimensions?: { width: number; height: number };
  colorPalette?: string[];
  accessibility?: AccessibilityScore;
}

export interface Issue {
  type: 'error' | 'warning' | 'info';
  message: string;
  line?: number;
  severity: number;
}

export interface AccessibilityScore {
  overall: number;
  contrast: number;
  structure: number;
  navigation: number;
}

export interface ProjectContext {
  id: string;
  name: string;
  description: string;
  goals: string[];
  type: 'web-app' | 'mobile-app' | 'desktop-app' | 'api' | 'other';
  technologies: string[];
  files: FileAttachment[];
  agents: Agent[];
  progress: ProjectProgress;
  createdAt: Date;
  updatedAt: Date;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  status: 'active' | 'paused' | 'archived';
  context?: {
    goals: string[];
    completedTasks?: number;
    activeTasks?: number;
  };
  createdAt: Date;
  updatedAt: Date;
}

export interface ProjectProgress {
  overall: number;
  phases: {
    planning: number;
    design: number;
    development: number;
    testing: number;
    deployment: number;
  };
  currentPhase: keyof ProjectProgress['phases'];
  milestones: Milestone[];
}

export interface Milestone {
  id: string;
  title: string;
  description: string;
  completed: boolean;
  dueDate?: Date;
  completedDate?: Date;
  assignedAgents: string[];
}

export type TaskStatus = 'pending' | 'in-progress' | 'completed' | 'blocked';
export type TaskPriority = 'low' | 'medium' | 'high';

export interface Task {
  id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  priority: TaskPriority;
  assignedAgent?: string; // agent id
  tags?: string[];
  dueDate?: Date;
  createdAt: Date;
  updatedAt: Date;
  completedAt?: Date;
}

export interface CollaborationTask {
  id: string;
  title: string;
  description: string;
  type: 'analysis' | 'design' | 'development' | 'review' | 'planning';
  status: 'pending' | 'in-progress' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  assignedAgents: string[];
  results?: TaskResult[];
  startedAt?: Date;
  completedAt?: Date;
  estimatedDuration: number; // in minutes
}

export interface TaskResult {
  agentId: string;
  contribution: string;
  quality: number;
  timestamp: Date;
  insights: string[];
}

export interface SystemMetrics {
  agentCount: number;
  activeAgents: number;
  collaborationQuality: number;
  systemCoherence: number;
  taskQueue: number;
  averageResponseTime: number;
  uptime: number;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  notifications: {
    agentUpdates: boolean;
    taskCompletions: boolean;
    systemAlerts: boolean;
    fileProcessing: boolean;
  };
  chatSettings: {
    autoScroll: boolean;
    showTimestamps: boolean;
    showAgentTypes: boolean;
    messageGrouping: boolean;
  };
  workflowSettings: {
    autoStartTasks: boolean;
    parallelProcessing: boolean;
    qualityThreshold: number;
    defaultAgentRoles: string[];
  };
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  timestamp: Date;
}

// WebSocket Event Types
export interface WebSocketEvent {
  type: string;
  data: any;
  timestamp: Date;
}

export interface AgentStatusEvent extends WebSocketEvent {
  type: 'agent_status';
  data: {
    agentId: string;
    status: 'online' | 'offline' | 'busy';
    activity: string;
  };
}

export interface TaskProgressEvent extends WebSocketEvent {
  type: 'task_progress';
  data: {
    taskId: string;
    progress: number;
    currentStep: string;
    agentUpdates: {
      agentId: string;
      status: string;
      contribution?: string;
    }[];
  };
}

export interface FileProcessingEvent extends WebSocketEvent {
  type: 'file_processing';
  data: {
    fileId: string;
    status: 'processing' | 'completed' | 'failed';
    progress: number;
    analysis?: FileAnalysis;
  };
}

// Drag and Drop Types
export interface DropItem {
  type: 'file' | 'folder';
  name: string;
  path: string;
  size?: number;
  files?: File[];
  isDirectory?: boolean;
}

export interface UploadProgress {
  fileId: string;
  fileName: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  error?: string;
}

// Component Props Types
export interface ChatInputProps {
  onSendMessage: (message: string, files?: File[]) => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
}

export interface FileDropZoneProps {
  onFileDrop: (files: File[]) => void;
  onFolderDrop?: (files: File[]) => void;
  acceptedTypes?: string[];
  maxSize?: number;
  multiple?: boolean;
  className?: string;
}

export interface AgentStatusPanelProps {
  agents: Agent[];
  onAgentSelect?: (agent: Agent) => void;
  selectedAgent?: Agent;
}

export interface ProjectContextPanelProps {
  context: ProjectContext;
  onContextUpdate: (context: Partial<ProjectContext>) => void;
  onGoalAdd: (goal: string) => void;
  onGoalRemove: (index: number) => void;
}

// Store Types (for Zustand)
export interface ChatStore {
  messages: ChatMessage[];
  currentContext: ProjectContext | null;
  agents: Agent[];
  systemMetrics: SystemMetrics;
  isConnected: boolean;
  isLoading: boolean;
  
  // Actions
  sendMessage: (content: string, files?: File[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateContext: (context: Partial<ProjectContext>) => void;
  updateAgents: (agents: Agent[]) => void;
  updateMetrics: (metrics: SystemMetrics) => void;
  setConnectionStatus: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
}

export interface FileStore {
  uploadingFiles: UploadProgress[];
  processedFiles: FileAttachment[];
  
  // Actions
  addUpload: (file: File) => void;
  updateUploadProgress: (fileId: string, progress: number) => void;
  completeUpload: (fileId: string, attachment: FileAttachment) => void;
  removeUpload: (fileId: string) => void;
}

// API Endpoint Types
export interface KairosAPI {
  // Chat endpoints
  sendMessage: (message: string, context?: string, files?: File[]) => Promise<APIResponse<ChatMessage>>;
  getMessages: (limit?: number, offset?: number) => Promise<APIResponse<ChatMessage[]>>;
  
  // Agent endpoints
  getAgents: () => Promise<APIResponse<Agent[]>>;
  getAgent: (id: string) => Promise<APIResponse<Agent>>;
  createCollaborationTask: (task: Omit<CollaborationTask, 'id' | 'results' | 'startedAt' | 'completedAt'>) => Promise<APIResponse<CollaborationTask>>;
  
  // File endpoints
  uploadFile: (file: File, context?: string) => Promise<APIResponse<FileAttachment>>;
  analyzeFile: (fileId: string) => Promise<APIResponse<FileAnalysis>>;
  getFileAnalysis: (fileId: string) => Promise<APIResponse<FileAnalysis>>;
  
  // Project endpoints
  createProject: (project: Omit<ProjectContext, 'id' | 'createdAt' | 'updatedAt'>) => Promise<APIResponse<ProjectContext>>;
  updateProject: (id: string, updates: Partial<ProjectContext>) => Promise<APIResponse<ProjectContext>>;
  getProject: (id: string) => Promise<APIResponse<ProjectContext>>;
  
  // System endpoints
  getSystemMetrics: () => Promise<APIResponse<SystemMetrics>>;
  getSystemHealth: () => Promise<APIResponse<{ status: string; uptime: number; }>>;
}