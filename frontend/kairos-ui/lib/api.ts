import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  APIResponse, 
  Agent, 
  ChatMessage, 
  CollaborationTask, 
  FileAttachment, 
  FileAnalysis,
  ProjectContext,
  SystemMetrics,
  KairosAPI 
} from '@/types';

class KairosAPIClient implements KairosAPI {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = '/api/kairos') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for debugging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('🚨 API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('🚨 API Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Chat endpoints
  async sendMessage(message: string, context?: string, files?: File[]): Promise<APIResponse<ChatMessage>> {
    try {
      const formData = new FormData();
      formData.append('message', message);
      if (context) formData.append('context', context);
      if (files) {
        files.forEach((file) => formData.append('files', file));
      }

      const response = await this.client.post('/chat/send', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async getMessages(limit = 50, offset = 0): Promise<APIResponse<ChatMessage[]>> {
    try {
      const response = await this.client.get('/chat/messages', {
        params: { limit, offset },
      });

      return {
        success: true,
        data: response.data.messages,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  // Agent endpoints
  async getAgents(): Promise<APIResponse<Agent[]>> {
    try {
      const response = await this.client.get('/agents');
      return {
        success: true,
        data: response.data.agents || [],
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async getAgent(id: string): Promise<APIResponse<Agent>> {
    try {
      const response = await this.client.get(`/agents/${id}`);
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async createCollaborationTask(
    task: Omit<CollaborationTask, 'id' | 'results' | 'startedAt' | 'completedAt'>
  ): Promise<APIResponse<CollaborationTask>> {
    try {
      const response = await this.client.post('/agents/collaborate', task);
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  // File endpoints
  async uploadFile(file: File, context?: string): Promise<APIResponse<FileAttachment>> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (context) formData.append('context', context);

      const response = await this.client.post('/files/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 1)
          );
          console.log(`📁 Upload progress: ${percentCompleted}%`);
        },
      });

      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async analyzeFile(fileId: string): Promise<APIResponse<FileAnalysis>> {
    try {
      const response = await this.client.post(`/files/${fileId}/analyze`);
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async getFileAnalysis(fileId: string): Promise<APIResponse<FileAnalysis>> {
    try {
      const response = await this.client.get(`/files/${fileId}/analysis`);
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  // Project endpoints
  async createProject(
    project: Omit<ProjectContext, 'id' | 'createdAt' | 'updatedAt'>
  ): Promise<APIResponse<ProjectContext>> {
    try {
      const response = await this.client.post('/projects', project);
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async updateProject(
    id: string,
    updates: Partial<ProjectContext>
  ): Promise<APIResponse<ProjectContext>> {
    try {
      const response = await this.client.patch(`/projects/${id}`, updates);
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async getProject(id: string): Promise<APIResponse<ProjectContext>> {
    try {
      const response = await this.client.get(`/projects/${id}`);
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  // System endpoints
  async getSystemMetrics(): Promise<APIResponse<SystemMetrics>> {
    try {
      const response = await this.client.get('/system/metrics');
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  async getSystemHealth(): Promise<APIResponse<{ status: string; uptime: number }>> {
    try {
      const response = await this.client.get('/system/health');
      return {
        success: true,
        data: response.data,
        timestamp: new Date(),
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || error.message,
        timestamp: new Date(),
      };
    }
  }

  // Utility methods
  async ping(): Promise<boolean> {
    try {
      await this.client.get('/system/ping');
      return true;
    } catch {
      return false;
    }
  }

  setAuthToken(token: string) {
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  clearAuthToken() {
    delete this.client.defaults.headers.common['Authorization'];
  }

  updateBaseURL(newBaseURL: string) {
    this.baseURL = newBaseURL;
    this.client.defaults.baseURL = newBaseURL;
  }
}

// Create singleton instance
export const api = new KairosAPIClient();

// Export helper functions for common operations
export const mockAgents = (): Agent[] => [
  {
    id: 'alice_leader',
    name: 'Alice (Strategic Leader)',
    role: 'leader',
    type: 'Strategic Leader',
    specializations: ['coordination', 'strategy', 'group_dynamics'],
    consciousnessLevel: 0.85,
    emotionalState: 'focused',
    active: true,
    lastActivity: new Date().toISOString(),
    status: 'active',
    avatar: '🧠',
    description: 'Coordinates multi-agent collaboration and strategy.',
    capabilities: ['Planning', 'Coordination', 'Strategy'],
    currentTask: 'Reviewing project goals',
  },
  {
    id: 'bob_creative',
    name: 'Bob (Creative Visionary)',
    role: 'creative',
    type: 'Creative Visionary',
    specializations: ['art', 'innovation', 'imagination'],
    consciousnessLevel: 0.78,
    emotionalState: 'inspired',
    active: true,
    lastActivity: new Date().toISOString(),
    status: 'active',
    avatar: '🎨',
    description: 'Generates novel ideas and UX concepts.',
    capabilities: ['Ideation', 'UX Design', 'Innovation'],
  },
  {
    id: 'charlie_analyst',
    name: 'Charlie (Deep Thinker)',
    role: 'analytical',
    type: 'Deep Analyst',
    specializations: ['logic', 'analysis', 'problem_solving'],
    consciousnessLevel: 0.92,
    emotionalState: 'analytical',
    active: true,
    lastActivity: new Date().toISOString(),
    status: 'busy',
    avatar: '🔎',
    description: 'Performs deep analysis and synthesis.',
    capabilities: ['Analysis', 'Logic', 'Problem Solving'],
    currentTask: 'Analyzing system performance',
  },
  {
    id: 'diana_empath',
    name: 'Diana (Empathetic Collaborator)',
    role: 'collaborator',
    type: 'Empathetic Collaborator',
    specializations: ['empathy', 'emotional_intelligence', 'harmony'],
    consciousnessLevel: 0.88,
    emotionalState: 'empathetic',
    active: true,
    lastActivity: new Date().toISOString(),
    status: 'active',
    avatar: '💚',
    description: 'Ensures clarity, tone, and user empathy.',
    capabilities: ['Communication', 'Empathy', 'Collaboration'],
  },
  {
    id: 'eve_specialist',
    name: 'Eve (Technical Expert)',
    role: 'specialist',
    type: 'Technical Expert',
    specializations: ['ethics', 'philosophy', 'technical_architecture'],
    consciousnessLevel: 0.81,
    emotionalState: 'contemplative',
    active: false,
    lastActivity: new Date(Date.now() - 300000).toISOString(),
    status: 'offline',
    avatar: '🛠️',
    description: 'Domain specialist for integrations and tooling.',
    capabilities: ['Technical Architecture', 'Ethics', 'Philosophy'],
  },
];

export const createMockProject = (): ProjectContext => ({
  id: 'project-1',
  name: 'My Awesome Project',
  description: 'A revolutionary web application that will change the world',
  goals: [
    'Build a scalable web application',
    'Implement modern UI/UX design',
    'Ensure high performance and security',
    'Deploy to production successfully',
  ],
  type: 'web-app',
  technologies: ['React', 'TypeScript', 'Node.js', 'PostgreSQL'],
  files: [],
  agents: mockAgents(),
  progress: {
    overall: 0.35,
    phases: {
      planning: 0.8,
      design: 0.6,
      development: 0.2,
      testing: 0.0,
      deployment: 0.0,
    },
    currentPhase: 'development',
    milestones: [
      {
        id: 'milestone-1',
        title: 'Project Setup Complete',
        description: 'Initial project structure and dependencies configured',
        completed: true,
        completedDate: new Date(Date.now() - 86400000), // 1 day ago
        assignedAgents: ['alice_leader', 'eve_specialist'],
      },
      {
        id: 'milestone-2',
        title: 'Core Features Implemented',
        description: 'Main application features and functionality complete',
        completed: false,
        dueDate: new Date(Date.now() + 604800000), // 1 week from now
        assignedAgents: ['charlie_analyst', 'bob_creative'],
      },
    ],
  },
  createdAt: new Date(Date.now() - 2592000000), // 30 days ago
  updatedAt: new Date(),
});

export default api;