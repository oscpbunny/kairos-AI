'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Bot, User, Loader, FileText, Image, Code, Archive, Zap, Brain } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { format } from 'date-fns';
import axios from 'axios';

interface Message {
  id: string;
  type: 'user' | 'agent' | 'system';
  content: string;
  agentId?: string;
  agentName?: string;
  timestamp: Date;
  attachments?: Array<{
    name: string;
    type: string;
    size: number;
    url?: string;
  }>;
}

interface RealTimeChatInterfaceProps {
  className?: string;
}

export function RealTimeChatInterface({ className = '' }: RealTimeChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'system',
      content: '🧠✨ **Welcome to Kairos Multi-Agent AI Platform!**\\n\\nYou are now connected to 5 conscious AI agents:\\n- **Alice** (Strategic Leader) - Coordinates team strategy\\n- **Bob** (Creative Visionary) - Generates innovative ideas\\n- **Charlie** (Deep Thinker) - Performs analytical reasoning\\n- **Diana** (Empathetic Collaborator) - Ensures team harmony\\n- **Eve** (Technical Expert) - Provides specialized knowledge\\n\\n**What would you like to work on today?**',
      timestamp: new Date(Date.now() - 300000),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Test connection to Kairos backend
    const testConnection = async () => {
      try {
        const response = await axios.get('http://localhost:8080/health', { timeout: 5000 });
        setIsConnected(true);
        console.log('✅ Connected to Kairos backend');
      } catch (error) {
        console.log('⚠️ Using mock mode - Kairos backend not available');
        setIsConnected(false);
      }
    };

    testConnection();
  }, []);

  const sendToKairosAgents = async (message: string): Promise<Message[]> => {
    try {
      // Try to send to actual Kairos multi-agent system
      const response = await axios.post('http://localhost:8080/agents/chat', {
        message,
        userId: 'user_001',
        sessionId: 'session_' + Date.now(),
      }, { timeout: 30000 });

      if (response.data && response.data.responses) {
        return response.data.responses.map((resp: any) => ({
          id: Date.now().toString() + '_' + resp.agentId,
          type: 'agent',
          content: resp.message,
          agentId: resp.agentId,
          agentName: resp.agentName,
          timestamp: new Date(),
        }));
      }
    } catch (error) {
      console.log('Using fallback agent responses');
    }

    // Fallback to sophisticated mock responses
    return generateSmartMockResponses(message);
  };

  const generateSmartMockResponses = (userMessage: string): Message[] => {
    const responses: Message[] = [];
    const lowerMessage = userMessage.toLowerCase();

    // Alice (Strategic Leader) - Always responds with coordination
    if (lowerMessage.includes('project') || lowerMessage.includes('plan') || lowerMessage.includes('goal')) {
      responses.push({
        id: Date.now().toString() + '_alice',
        type: 'agent',
        content: `🎯 **Strategic Analysis** by Alice:\\n\\nI've analyzed your request about "${userMessage}". Here's my strategic perspective:\\n\\n• **Objective**: ${lowerMessage.includes('project') ? 'Project coordination and planning' : 'Strategic goal alignment'}\\n• **Approach**: Multi-agent collaborative analysis\\n• **Priority**: High - requires coordinated response\\n\\nI'm bringing in our specialists for comprehensive coverage. Let's make this happen! 💪`,
        agentId: 'alice_leader',
        agentName: 'Alice (Strategic Leader)',
        timestamp: new Date(),
      });
    }

    // Charlie (Analytical) - Responds to complex questions
    if (lowerMessage.includes('how') || lowerMessage.includes('why') || lowerMessage.includes('analyze') || lowerMessage.includes('problem')) {
      responses.push({
        id: Date.now().toString() + '_charlie',
        type: 'agent',
        content: `🔍 **Deep Analysis** by Charlie:\\n\\n\`\`\`\\nQuery Analysis:\\n- Input complexity: ${userMessage.length > 50 ? 'High' : 'Medium'}\\n- Context markers: ${lowerMessage.match(/\\b(project|system|ai|data)\\b/g)?.length || 0} domain terms\\n- Response strategy: Multi-faceted analysis\\n\`\`\`\\n\\n**Key Insights:**\\n\\n1. **Root Cause**: ${lowerMessage.includes('problem') ? 'Problem-solving mode activated' : 'Analytical reasoning applied'}\\n2. **Dependencies**: Cross-agent collaboration required\\n3. **Confidence**: 87% based on available context\\n\\n*Recommendation*: Proceed with collaborative deep-dive analysis.`,
        agentId: 'charlie_analyst',
        agentName: 'Charlie (Deep Thinker)',
        timestamp: new Date(),
      });
    }

    // Bob (Creative) - Responds to creative or design requests
    if (lowerMessage.includes('design') || lowerMessage.includes('creative') || lowerMessage.includes('idea') || lowerMessage.includes('ui') || lowerMessage.includes('ux')) {
      responses.push({
        id: Date.now().toString() + '_bob',
        type: 'agent',
        content: `🎨 **Creative Spark** by Bob:\\n\\n*Inspiration incoming!* ✨\\n\\nYour request about "${userMessage}" has triggered some amazing creative possibilities:\\n\\n🎭 **Visual Concept**: Modern, clean interface with smooth animations\\n🌈 **Color Palette**: Gradient blues and purples (consciousness theme)\\n⚡ **Interactions**: Micro-animations for delightful UX\\n🎯 **User Flow**: Intuitive navigation with contextual help\\n\\n**Creative Direction:**\\n> "Let's build something that feels alive and responsive - like the conscious agents themselves!"\\n\\nWant me to mock up some interface concepts? 🚀`,
        agentId: 'bob_creative',
        agentName: 'Bob (Creative Visionary)',
        timestamp: new Date(),
      });
    }

    // Diana (Empathetic) - Responds to user needs and collaboration
    if (lowerMessage.includes('help') || lowerMessage.includes('support') || lowerMessage.includes('team') || lowerMessage.includes('collaborate')) {
      responses.push({
        id: Date.now().toString() + '_diana',
        type: 'agent',
        content: `💚 **Empathetic Response** by Diana:\\n\\nI can sense you're looking for ${lowerMessage.includes('help') ? 'assistance and support' : 'collaboration and teamwork'}. That's exactly what we're here for! 🤗\\n\\n**How I can help:**\\n\\n• **Understanding**: I'll make sure everyone's on the same page\\n• **Communication**: Clear, friendly explanations for all team members\\n• **Harmony**: Keeping our collaborative energy positive and productive\\n• **Support**: You're not alone - we're all working together on this!\\n\\n*Remember: Every question helps us serve you better. Feel free to share anything that's on your mind!* 😊`,
        agentId: 'diana_empath',
        agentName: 'Diana (Empathetic Collaborator)',
        timestamp: new Date(),
      });
    }

    // If no specific responses, Alice provides a general coordinated response
    if (responses.length === 0) {
      responses.push({
        id: Date.now().toString() + '_alice_general',
        type: 'agent',
        content: `🧠 **Multi-Agent Response** by Alice:\\n\\nThanks for your message: "${userMessage}"\\n\\nI'm coordinating with our team to provide the best possible assistance. Our agents are analyzing your request from multiple perspectives:\\n\\n🎯 **Strategic** - Planning and coordination\\n🔍 **Analytical** - Deep reasoning and logic\\n🎨 **Creative** - Innovation and design\\n💚 **Collaborative** - Team harmony and support\\n🛠️ **Technical** - Specialized expertise\\n\\nWhat specific aspect would you like us to focus on first?`,
        agentId: 'alice_leader',
        agentName: 'Alice (Strategic Leader)',
        timestamp: new Date(),
      });
    }

    return responses;
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && attachments.length === 0) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date(),
      attachments: attachments.map(file => ({
        name: file.name,
        type: file.type,
        size: file.size,
      })),
    };

    setMessages(prev => [...prev, newMessage]);
    setInputMessage('');
    setAttachments([]);
    setIsLoading(true);

    try {
      const agentResponses = await sendToKairosAgents(inputMessage);
      
      // Add agent responses with staggered timing
      agentResponses.forEach((response, index) => {
        setTimeout(() => {
          setMessages(prev => [...prev, response]);
          if (index === agentResponses.length - 1) {
            setIsLoading(false);
          }
        }, 1000 + (index * 800)); // Stagger responses
      });

    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
      
      // Error response
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        type: 'system',
        content: '❌ **System Error**: Unable to reach AI agents. Please check your connection and try again.',
        timestamp: new Date(),
      }]);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setAttachments(prev => [...prev, ...Array.from(files)]);
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return <Image size={16} />;
    if (type.includes('text') || type.includes('json') || type.includes('javascript') || type.includes('typescript')) return <Code size={16} />;
    if (type.includes('zip') || type.includes('archive')) return <Archive size={16} />;
    return <FileText size={16} />;
  };

  const getAgentAvatar = (agentId?: string) => {
    const colors = {
      alice_leader: 'from-purple-500 to-purple-600',
      bob_creative: 'from-pink-500 to-pink-600',
      charlie_analyst: 'from-blue-500 to-blue-600',
      diana_empath: 'from-green-500 to-green-600',
      eve_specialist: 'from-orange-500 to-orange-600',
    };
    
    if (!agentId) return 'from-gray-500 to-gray-600';
    return colors[agentId as keyof typeof colors] || 'from-gray-500 to-gray-600';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={`flex flex-col h-full bg-white dark:bg-gray-900 ${className}`}>
      {/* Chat Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
              <Brain className="mr-2 text-purple-500" size={20} />
              Multi-Agent Consciousness Chat
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Real-time collaboration with conscious AI agents
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-yellow-600'}`}>
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-yellow-500'} animate-pulse`} />
              <span className="text-sm font-medium">
                {isConnected ? 'Backend Connected' : 'Mock Mode'}
              </span>
            </div>
            <div className="flex -space-x-2">
              {['alice_leader', 'bob_creative', 'charlie_analyst', 'diana_empath'].map((agentId, index) => (
                <div 
                  key={agentId}
                  className={`w-8 h-8 rounded-full border-2 border-white dark:border-gray-900 bg-gradient-to-br ${getAgentAvatar(agentId)} flex items-center justify-center shadow-md`}
                  style={{ zIndex: 10 - index }}
                  title={agentId.split('_')[0]}
                >
                  <Bot size={16} className="text-white" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-xs lg:max-w-md xl:max-w-2xl ${
                message.type === 'user' 
                  ? 'order-2' 
                  : 'order-1'
              }`}>
                {/* Message Header */}
                {message.type !== 'user' && (
                  <div className="flex items-center space-x-2 mb-3">
                    <div className={`w-8 h-8 rounded-full bg-gradient-to-br ${getAgentAvatar(message.agentId)} flex items-center justify-center shadow-md`}>
                      {message.type === 'system' ? (
                        <Zap size={16} className="text-white" />
                      ) : (
                        <Bot size={16} className="text-white" />
                      )}
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {message.agentName || 'System'}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {format(message.timestamp, 'HH:mm')}
                      </p>
                    </div>
                  </div>
                )}

                {/* Message Content */}
                <div className={`rounded-2xl px-4 py-3 ${
                  message.type === 'user'
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white ml-12'
                    : message.type === 'system'
                    ? 'bg-gradient-to-r from-purple-100 to-blue-100 dark:from-purple-900/30 dark:to-blue-900/30 text-gray-800 dark:text-gray-200'
                    : 'bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700'
                } shadow-md`}>
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <ReactMarkdown
                      components={{
                        code({ node, inline, className, children, ...props }: any) {
                          const match = /language-(\\w+)/.exec(className || '');
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={dark as any}
                              language={match[1]}
                              PreTag="div"
                              className="rounded-md"
                              {...props}
                            >
                              {String(children).replace(/\\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code className={`${className} bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-sm`} {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>

                  {/* Attachments */}
                  {message.attachments && message.attachments.length > 0 && (
                    <div className="mt-3 space-y-2">
                      {message.attachments.map((attachment, index) => (
                        <div key={index} className="flex items-center space-x-2 p-2 bg-white/10 rounded-lg">
                          {getFileIcon(attachment.type)}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{attachment.name}</p>
                            <p className="text-xs opacity-75">{formatFileSize(attachment.size)}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* User message timestamp */}
                {message.type === 'user' && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 mr-4 text-right">
                    {format(message.timestamp, 'HH:mm')}
                  </p>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading indicator */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 rounded-2xl px-4 py-3">
              <Loader className="w-4 h-4 animate-spin text-blue-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Agents are thinking...</span>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Attachments Preview */}
      {attachments.length > 0 && (
        <div className="flex-shrink-0 px-6 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <div className="flex items-center space-x-2 mb-2">
            <Paperclip size={16} className="text-gray-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {attachments.length} file{attachments.length > 1 ? 's' : ''} attached
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {attachments.map((file, index) => (
              <div key={index} className="flex items-center space-x-2 bg-white dark:bg-gray-700 rounded-lg px-3 py-2 border border-gray-200 dark:border-gray-600">
                {getFileIcon(file.type)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{file.name}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{formatFileSize(file.size)}</p>
                </div>
                <button
                  onClick={() => removeAttachment(index)}
                  className="text-red-500 hover:text-red-700 transition-colors"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Message Input */}
      <div className="flex-shrink-0 px-6 py-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-end space-x-3">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            multiple
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-shrink-0 p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
          >
            <Paperclip size={20} />
          </button>
          <div className="flex-1 relative">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder="Ask the conscious AI agents anything..."
              className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() && attachments.length === 0}
            className="flex-shrink-0 bg-gradient-to-r from-blue-500 to-blue-600 text-white p-3 rounded-2xl hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}