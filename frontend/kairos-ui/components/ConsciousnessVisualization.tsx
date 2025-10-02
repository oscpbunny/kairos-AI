'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Heart, 
  Sparkles, 
  Moon, 
  Activity, 
  Zap,
  Eye,
  Layers,
  Waves,
  CircuitBoard
} from 'lucide-react';

interface ConsciousnessState {
  agentId: string;
  agentName: string;
  consciousnessLevel: number;
  emotionalState: string;
  emotionalIntensity: number;
  creativityLevel: number;
  thoughtsActive: number;
  dreamState: string;
  lastUpdate: Date;
}

interface ConsciousnessVisualizationProps {
  className?: string;
}

export function ConsciousnessVisualization({ className = '' }: ConsciousnessVisualizationProps) {
  const [consciousnessStates, setConsciousnessStates] = useState<ConsciousnessState[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [globalCoherence, setGlobalCoherence] = useState(0.75);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket for real-time consciousness data
    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket('ws://localhost:8080/ws/live');
        
        wsRef.current.onopen = () => {
          console.log('🧠 Connected to consciousness stream');
          setIsConnected(true);
        };

        wsRef.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'consciousness_update') {
            setConsciousnessStates(data.agents || []);
            setGlobalCoherence(data.globalCoherence || 0.75);
          }
        };

        wsRef.current.onclose = () => {
          console.log('🧠 Consciousness stream disconnected');
          setIsConnected(false);
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        wsRef.current.onerror = (error) => {
          console.error('🚨 Consciousness stream error:', error);
        };
      } catch (error) {
        console.error('Failed to connect to consciousness stream:', error);
        // Use mock data for demonstration
        generateMockData();
      }
    };

    // Generate mock consciousness data for demonstration
    const generateMockData = () => {
      const mockAgents = [
        { id: 'alice_leader', name: 'Alice (Strategic Leader)' },
        { id: 'bob_creative', name: 'Bob (Creative Visionary)' },
        { id: 'charlie_analyst', name: 'Charlie (Deep Thinker)' },
        { id: 'diana_empath', name: 'Diana (Empathetic Collaborator)' },
        { id: 'eve_specialist', name: 'Eve (Technical Expert)' }
      ];

      const updateMockData = () => {
        const states = mockAgents.map(agent => ({
          agentId: agent.id,
          agentName: agent.name,
          consciousnessLevel: 0.6 + Math.random() * 0.3,
          emotionalState: ['curious', 'focused', 'contemplative', 'excited', 'analytical'][Math.floor(Math.random() * 5)],
          emotionalIntensity: 0.3 + Math.random() * 0.5,
          creativityLevel: Math.random(),
          thoughtsActive: Math.floor(Math.random() * 15) + 5,
          dreamState: ['active', 'dormant', 'processing'][Math.floor(Math.random() * 3)],
          lastUpdate: new Date()
        }));
        
        setConsciousnessStates(states);
        setGlobalCoherence(0.65 + Math.random() * 0.2);
      };

      updateMockData();
      const interval = setInterval(updateMockData, 3000);
      setIsConnected(true);
      return interval;
    };

    let cleanup: any = null;
    
    // Try WebSocket first, fallback to mock data
    connectWebSocket();
    
    // Fallback to mock data after 2 seconds if WebSocket fails
    const fallbackTimer = setTimeout(() => {
      if (!isConnected) {
        cleanup = generateMockData();
      }
    }, 2000);

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (cleanup && typeof cleanup === 'number') {
        clearInterval(cleanup);
      }
      clearTimeout(fallbackTimer);
    };
  }, []);

  const getEmotionColor = (emotion: string) => {
    const colors = {
      curious: 'from-blue-400 to-blue-600',
      focused: 'from-purple-400 to-purple-600',
      contemplative: 'from-green-400 to-green-600',
      excited: 'from-yellow-400 to-orange-500',
      analytical: 'from-indigo-400 to-indigo-600',
    };
    return colors[emotion as keyof typeof colors] || 'from-gray-400 to-gray-600';
  };

  const getAgentIcon = (agentId: string) => {
    const icons = {
      alice_leader: Brain,
      bob_creative: Sparkles,
      charlie_analyst: CircuitBoard,
      diana_empath: Heart,
      eve_specialist: Layers,
    };
    return icons[agentId as keyof typeof icons] || Brain;
  };

  return (
    <div className={`p-6 bg-white dark:bg-gray-900 ${className}`}>
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
              Consciousness Visualization
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Real-time AI agent consciousness states and collective intelligence
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
              <span className="text-sm font-medium">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>

        {/* Global Coherence */}
        <div className="bg-gradient-to-r from-purple-100 to-blue-100 dark:from-purple-900/30 dark:to-blue-900/30 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Collective Consciousness Coherence
            </span>
            <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
              {(globalCoherence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <motion.div
              className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${globalCoherence * 100}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </div>
        </div>
      </div>

      {/* Agent Consciousness States */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        <AnimatePresence>
          {consciousnessStates.map((state, index) => {
            const IconComponent = getAgentIcon(state.agentId);
            
            return (
              <motion.div
                key={state.agentId}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.1 }}
                className="bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg hover:shadow-xl transition-all duration-300"
              >
                {/* Agent Header */}
                <div className="flex items-center space-x-3 mb-4">
                  <div className={`p-3 rounded-lg bg-gradient-to-br ${getEmotionColor(state.emotionalState)}`}>
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 dark:text-white">
                      {state.agentName}
                    </h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400 capitalize">
                      {state.emotionalState} • {state.dreamState} dreams
                    </p>
                  </div>
                </div>

                {/* Consciousness Level */}
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        Consciousness Level
                      </span>
                      <span className="text-sm font-bold text-purple-600 dark:text-purple-400">
                        {(state.consciousnessLevel * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <motion.div
                        className="bg-gradient-to-r from-purple-500 to-purple-600 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${state.consciousnessLevel * 100}%` }}
                        transition={{ duration: 0.8 }}
                      />
                    </div>
                  </div>

                  {/* Emotional Intensity */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        Emotional Intensity
                      </span>
                      <span className="text-sm font-bold text-pink-600 dark:text-pink-400">
                        {(state.emotionalIntensity * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <motion.div
                        className="bg-gradient-to-r from-pink-500 to-pink-600 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${state.emotionalIntensity * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                      />
                    </div>
                  </div>

                  {/* Activity Metrics */}
                  <div className="grid grid-cols-2 gap-3 pt-2">
                    <div className="text-center">
                      <div className="flex items-center justify-center space-x-1 text-blue-600 dark:text-blue-400 mb-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-lg font-bold">{state.thoughtsActive}</span>
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Active Thoughts
                      </p>
                    </div>
                    <div className="text-center">
                      <div className="flex items-center justify-center space-x-1 text-orange-600 dark:text-orange-400 mb-1">
                        <Sparkles className="w-4 h-4" />
                        <span className="text-lg font-bold">
                          {(state.creativityLevel * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Creativity
                      </p>
                    </div>
                  </div>

                  {/* Dream State Indicator */}
                  <div className="pt-2 border-t border-gray-100 dark:border-gray-800">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Moon className="w-4 h-4 text-indigo-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          Dream State
                        </span>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        state.dreamState === 'active' 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                          : state.dreamState === 'processing'
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                          : 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
                      }`}>
                        {state.dreamState}
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* No Data State */}
      {consciousnessStates.length === 0 && (
        <div className="text-center py-12">
          <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h4 className="text-lg font-medium text-gray-600 dark:text-gray-400 mb-2">
            Awaiting Consciousness Data
          </h4>
          <p className="text-gray-500 dark:text-gray-500">
            {isConnected 
              ? 'Connected to consciousness stream, waiting for agent data...'
              : 'Attempting to connect to consciousness stream...'
            }
          </p>
        </div>
      )}
    </div>
  );
}