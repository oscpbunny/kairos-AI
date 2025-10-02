'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Bot, User, Loader, FileText, Image, Code, Archive } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vsDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { format } from 'date-fns';

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

interface ChatInterfaceProps {
  className?: string;
}

export function ChatInterface({ className = '' }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'system',
      content: 'Welcome to Kairos Multi-Agent AI Platform! 🤖✨\n\nYou can:\n- Chat with multiple specialized AI agents\n- Upload and analyze files\n- Set project context and goals\n- Coordinate collaborative tasks\n\nHow can we help you today?',
      timestamp: new Date(Date.now() - 300000),
    },
    {
      id: '2',
      type: 'agent',
      content: 'Hello! I\'m Alice, the strategic leader of this agent collective. I\'m here to help coordinate our team and understand your project goals. What kind of project are you working on?',
      agentId: 'alice_leader',
      agentName: 'Alice (Strategic Leader)',
      timestamp: new Date(Date.now() - 240000),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

    // Simulate agent responses
    setTimeout(() => {
      const agentResponses = [
        {
          agentId: 'charlie_analyst',
          agentName: 'Charlie (Deep Thinker)',
          content: `I've analyzed your message. Here are my insights:\n\n• **Key topics identified**: ${inputMessage.toLowerCase().includes('project') ? 'Project planning and strategy' : 'General inquiry'}\n• **Recommended approach**: Multi-agent collaboration\n• **Confidence level**: 85%\n\nWould you like me to coordinate with other agents for a comprehensive analysis?`,
        },
        {
          agentId: 'bob_creative',
          agentName: 'Bob (Creative Visionary)',
          content: `Creative perspective! 🎨\n\nI can help with:\n- Innovative solutions and brainstorming\n- UI/UX design concepts\n- Creative problem-solving approaches\n\n*What specific creative challenges are you facing?*`,
        },
      ];

      const randomResponse = agentResponses[Math.floor(Math.random() * agentResponses.length)];
      
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: randomResponse.content,
        agentId: randomResponse.agentId,
        agentName: randomResponse.agentName,
        timestamp: new Date(),
      }]);
      
      setIsLoading(false);
    }, 1500 + Math.random() * 1500);
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
      alice_leader: 'bg-purple-500',
      bob_creative: 'bg-pink-500',
      charlie_analyst: 'bg-blue-500',
      diana_empath: 'bg-green-500',
      eve_specialist: 'bg-orange-500',
    };
    
    if (!agentId) return 'bg-gray-500';
    return colors[agentId as keyof typeof colors] || 'bg-gray-500';
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
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Multi-Agent Chat</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">Collaborative AI conversation</p>
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex -space-x-2">
              {['alice_leader', 'bob_creative', 'charlie_analyst', 'diana_empath'].map((agentId, index) => (
                <div 
                  key={agentId}
                  className={`w-8 h-8 rounded-full border-2 border-white dark:border-gray-900 ${getAgentAvatar(agentId)} flex items-center justify-center`}
                  style={{ zIndex: 10 - index }}
                >
                  <Bot size={16} className="text-white" />
                </div>
              ))}
            </div>
            <span className="text-sm text-gray-500 dark:text-gray-400">4 agents online</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-xs lg:max-w-md xl:max-w-lg ${
                message.type === 'user' 
                  ? 'order-2' 
                  : 'order-1'
              }`}>
                {/* Message Header */}
                {message.type !== 'user' && (
                  <div className="flex items-center space-x-2 mb-2">
                    <div className={`w-6 h-6 rounded-full ${getAgentAvatar(message.agentId)} flex items-center justify-center`}>
                      {message.type === 'system' ? (
                        <Bot size={12} className="text-white" />
                      ) : (
                        <Bot size={12} className="text-white" />
                      )}
                    </div>
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {message.agentName || 'System'}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {format(message.timestamp, 'HH:mm')}
                    </span>
                  </div>
                )}

                {/* Message Bubble */}
                <div className={`rounded-lg px-4 py-3 ${
                  message.type === 'user'
                    ? 'bg-kairos-500 text-white ml-4'
                    : message.type === 'system'
                    ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
                    : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100'
                } shadow-sm`}>
                  {message.type === 'user' && (
                    <div className="flex items-center space-x-2 mb-2">
                      <User size={16} />
                      <span className="text-sm opacity-90">
                        {format(message.timestamp, 'HH:mm')}
                      </span>
                    </div>
                  )}
                  
                  <ReactMarkdown
                    className="prose prose-sm dark:prose-invert max-w-none"
                    components={{
                      code({node, inline, className, children, ...props}) {
                        const match = /language-(\\w+)/.exec(className || '');
                        return !inline && match ? (
                          <SyntaxHighlighter
                            style={vsDark}
                            language={match[1]}
                            PreTag="div"
                            className="rounded-md"
                            {...props}
                          >
                            {String(children).replace(/\\n$/, '')}
                          </SyntaxHighlighter>
                        ) : (
                          <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-sm" {...props}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>

                  {/* Attachments */}
                  {message.attachments && message.attachments.length > 0 && (
                    <div className="mt-3 space-y-2">
                      {message.attachments.map((attachment, index) => (
                        <div key={index} className="flex items-center space-x-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
                          {getFileIcon(attachment.type)}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{attachment.name}</p>
                            <p className="text-xs text-gray-500">{formatFileSize(attachment.size)}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading indicator */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start"
          >
            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg px-4 py-3 flex items-center space-x-2">
              <Loader className="animate-spin" size={16} />
              <span className="text-sm text-gray-600 dark:text-gray-400">Agent is thinking...</span>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-gray-200 dark:border-gray-700 p-4">
        {/* Attachments Preview */}
        {attachments.length > 0 && (
          <div className="mb-3 space-y-2">
            {attachments.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex items-center space-x-2">
                  {getFileIcon(file.type)}
                  <div>
                    <p className="text-sm font-medium">{file.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <button
                  onClick={() => removeAttachment(index)}
                  className="text-gray-400 hover:text-red-500"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input */}
        <div className="flex items-end space-x-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-shrink-0 p-2 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
          >
            <Paperclip size={20} />
          </button>
          
          <div className="flex-1 relative">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask the agents anything... You can also upload files!"
              className="w-full px-4 py-3 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white resize-none focus:ring-2 focus:ring-kairos-500 focus:border-transparent"
              rows={3}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
          </div>
          
          <button
            onClick={handleSendMessage}
            disabled={(!inputMessage.trim() && attachments.length === 0) || isLoading}
            className="flex-shrink-0 p-3 bg-kairos-500 text-white rounded-lg hover:bg-kairos-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={20} />
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileUpload}
          className="hidden"
          accept="*/*"
        />
      </div>
    </div>
  );
}