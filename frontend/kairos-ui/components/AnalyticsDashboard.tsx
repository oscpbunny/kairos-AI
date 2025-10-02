'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { api } from '@/lib/api';

interface Metric {
  label: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
}

interface ChartData {
  labels: string[];
  data: number[];
}

interface AnalyticsDashboardProps {
  className?: string;
}

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ className = '' }) => {
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [metrics, setMetrics] = useState<{
    system: Metric[];
    agents: Metric[];
    tasks: Metric[];
  }>({
    system: [],
    agents: [],
    tasks: []
  });
  const [chartData, setChartData] = useState<{
    taskCompletion: ChartData;
    agentActivity: ChartData;
    systemLoad: ChartData;
  }>({
    taskCompletion: { labels: [], data: [] },
    agentActivity: { labels: [], data: [] },
    systemLoad: { labels: [], data: [] }
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAnalytics();
  }, [timeRange]);

  const loadAnalytics = async () => {
    try {
      setError(null);
      setLoading(true);
      
      // Generate mock analytics data
      const mockMetrics = {
        system: [
          { label: 'Total Agents', value: 12, change: 2, trend: 'up' as const },
          { label: 'Active Tasks', value: 45, change: -3, trend: 'down' as const },
          { label: 'Completed Today', value: 28, change: 8, trend: 'up' as const },
          { label: 'Success Rate', value: 94.5, change: 1.2, trend: 'up' as const }
        ],
        agents: [
          { label: 'Average Response Time', value: 1.2, change: -0.3, trend: 'up' as const },
          { label: 'Agent Utilization', value: 78, change: 5, trend: 'up' as const },
          { label: 'Idle Agents', value: 3, change: -1, trend: 'up' as const },
          { label: 'Collaboration Score', value: 8.7, change: 0.4, trend: 'up' as const }
        ],
        tasks: [
          { label: 'Tasks Created', value: 156, change: 12, trend: 'up' as const },
          { label: 'Tasks Completed', value: 142, change: 8, trend: 'up' as const },
          { label: 'Average Time', value: 4.2, change: -0.5, trend: 'up' as const },
          { label: 'Blocked Tasks', value: 2, change: -1, trend: 'up' as const }
        ]
      };

      const generateTimeLabels = () => {
        const now = new Date();
        const labels: string[] = [];
        const points = timeRange === '1h' ? 12 : timeRange === '24h' ? 24 : timeRange === '7d' ? 7 : 30;
        
        for (let i = points - 1; i >= 0; i--) {
          const date = new Date(now);
          if (timeRange === '1h') {
            date.setMinutes(date.getMinutes() - (i * 5));
            labels.push(date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
          } else if (timeRange === '24h') {
            date.setHours(date.getHours() - i);
            labels.push(date.toLocaleTimeString([], { hour: '2-digit' }));
          } else if (timeRange === '7d') {
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString([], { weekday: 'short' }));
          } else {
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString([], { month: 'short', day: 'numeric' }));
          }
        }
        return labels;
      };

      const generateChartData = (base: number, variance: number, trend: number = 0) => {
        const labels = generateTimeLabels();
        const data = labels.map((_, i) => {
          const trendValue = trend * i;
          const randomVariance = (Math.random() - 0.5) * variance;
          return Math.max(0, base + trendValue + randomVariance);
        });
        return { labels, data };
      };

      const mockChartData = {
        taskCompletion: generateChartData(25, 10, 0.5),
        agentActivity: generateChartData(75, 15, 0.2),
        systemLoad: generateChartData(45, 20, -0.3)
      };

      setMetrics(mockMetrics);
      setChartData(mockChartData);
    } catch (err) {
      setError('Failed to load analytics');
      console.error('Error loading analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  const SimpleLineChart: React.FC<{ data: ChartData; color: string; title: string }> = ({ 
    data, color, title 
  }) => {
    const max = Math.max(...data.data);
    const min = Math.min(...data.data);
    const range = max - min || 1;

    return (
      <div className="card p-4">
        <h4 className="font-medium mb-4">{title}</h4>
        <div className="h-32 relative">
          <svg className="w-full h-full" viewBox="0 0 300 100">
            <defs>
              <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{ stopColor: color, stopOpacity: 0.3 }} />
                <stop offset="100%" style={{ stopColor: color, stopOpacity: 0 }} />
              </linearGradient>
            </defs>
            
            {/* Grid lines */}
            {[...Array(5)].map((_, i) => (
              <line
                key={i}
                x1="0"
                y1={i * 20}
                x2="300"
                y2={i * 20}
                stroke="currentColor"
                strokeOpacity="0.1"
                strokeWidth="1"
              />
            ))}
            
            {/* Data line */}
            <polyline
              fill="none"
              stroke={color}
              strokeWidth="2"
              points={data.data
                .map((value, index) => {
                  const x = (index / (data.data.length - 1)) * 300;
                  const y = 100 - ((value - min) / range) * 80 - 10;
                  return `${x},${y}`;
                })
                .join(' ')}
            />
            
            {/* Area fill */}
            <polygon
              fill={`url(#gradient-${color})`}
              points={[
                ...data.data.map((value, index) => {
                  const x = (index / (data.data.length - 1)) * 300;
                  const y = 100 - ((value - min) / range) * 80 - 10;
                  return `${x},${y}`;
                }),
                '300,90',
                '0,90'
              ].join(' ')}
            />
            
            {/* Data points */}
            {data.data.map((value, index) => {
              const x = (index / (data.data.length - 1)) * 300;
              const y = 100 - ((value - min) / range) * 80 - 10;
              return (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r="3"
                  fill={color}
                  className="opacity-75 hover:opacity-100 transition-opacity"
                />
              );
            })}
          </svg>
        </div>
        
        <div className="flex justify-between text-xs text-gray-500 mt-2">
          <span>{data.labels[0]}</span>
          <span>{data.labels[Math.floor(data.labels.length / 2)]}</span>
          <span>{data.labels[data.labels.length - 1]}</span>
        </div>
        
        <div className="text-center mt-2">
          <span className="text-2xl font-bold" style={{ color }}>
            {data.data[data.data.length - 1].toFixed(0)}
          </span>
          <span className="text-sm text-gray-500 ml-1">current</span>
        </div>
      </div>
    );
  };

  const MetricCard: React.FC<{ metric: Metric }> = ({ metric }) => {
    const trendIcon = metric.trend === 'up' ? '↗️' : metric.trend === 'down' ? '↘️' : '→';
    const trendColor = metric.trend === 'up' ? 'text-green-600' : metric.trend === 'down' ? 'text-red-600' : 'text-gray-600';

    return (
      <div className="card p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">{metric.label}</span>
          <span className={`text-sm ${trendColor}`}>
            {trendIcon}
          </span>
        </div>
        <div className="flex items-end justify-between">
          <span className="text-2xl font-bold">
            {metric.label.includes('Rate') || metric.label.includes('Score') 
              ? `${metric.value}%` 
              : metric.label.includes('Time')
              ? `${metric.value}s`
              : metric.value.toLocaleString()}
          </span>
          <span className={`text-xs ${trendColor}`}>
            {metric.change > 0 ? '+' : ''}{metric.change}
          </span>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-300 dark:bg-gray-700 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-300 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <div className="text-red-500 mb-2">⚠️ {error}</div>
        <button onClick={loadAnalytics} className="btn btn-primary">Retry</button>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Analytics Dashboard</h2>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500">Time Range:</span>
          <select
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-sm"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>
      </div>

      {/* System Metrics */}
      <div>
        <h3 className="text-lg font-medium mb-4 flex items-center">
          <span className="mr-2">🖥️</span>
          System Metrics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.system.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <MetricCard metric={metric} />
            </motion.div>
          ))}
        </div>
      </div>

      {/* Agent Metrics */}
      <div>
        <h3 className="text-lg font-medium mb-4 flex items-center">
          <span className="mr-2">🤖</span>
          Agent Performance
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.agents.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 0.4 }}
            >
              <MetricCard metric={metric} />
            </motion.div>
          ))}
        </div>
      </div>

      {/* Task Metrics */}
      <div>
        <h3 className="text-lg font-medium mb-4 flex items-center">
          <span className="mr-2">📋</span>
          Task Analytics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.tasks.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 0.8 }}
            >
              <MetricCard metric={metric} />
            </motion.div>
          ))}
        </div>
      </div>

      {/* Charts */}
      <div>
        <h3 className="text-lg font-medium mb-4 flex items-center">
          <span className="mr-2">📊</span>
          Performance Trends
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.2 }}
          >
            <SimpleLineChart
              data={chartData.taskCompletion}
              color="#3b82f6"
              title="Task Completion Rate"
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.4 }}
          >
            <SimpleLineChart
              data={chartData.agentActivity}
              color="#10b981"
              title="Agent Activity Level"
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.6 }}
          >
            <SimpleLineChart
              data={chartData.systemLoad}
              color="#f59e0b"
              title="System Load Average"
            />
          </motion.div>
        </div>
      </div>

      {/* Activity Feed */}
      <div>
        <h3 className="text-lg font-medium mb-4 flex items-center">
          <span className="mr-2">📈</span>
          Recent Activity
        </h3>
        <div className="card p-4">
          <div className="space-y-3">
            {[
              { time: '5 min ago', event: 'Agent "Data Analyst" completed task "Market Research"', type: 'success' },
              { time: '12 min ago', event: 'New project "E-commerce Platform" created', type: 'info' },
              { time: '18 min ago', event: 'System performance optimization completed', type: 'success' },
              { time: '25 min ago', event: 'Agent "QA Specialist" reported an issue', type: 'warning' },
              { time: '32 min ago', event: 'Task "Database Migration" assigned to "DevOps Agent"', type: 'info' },
              { time: '45 min ago', event: 'Weekly analytics report generated', type: 'info' },
            ].map((activity, index) => (
              <div key={index} className="flex items-start space-x-3 text-sm">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  activity.type === 'success' ? 'bg-green-500' :
                  activity.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                }`}></div>
                <div className="flex-1 min-w-0">
                  <p className="text-gray-900 dark:text-gray-100">{activity.event}</p>
                  <p className="text-gray-500 text-xs">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <button className="text-sm text-blue-600 hover:text-blue-700">
              View all activity →
            </button>
          </div>
        </div>
      </div>

      {/* Export and Actions */}
      <div className="flex justify-center space-x-4">
        <button className="btn btn-secondary">
          📊 Export Report
        </button>
        <button className="btn btn-secondary">
          📧 Schedule Email
        </button>
        <button className="btn btn-secondary">
          ⚙️ Configure Alerts
        </button>
      </div>
    </div>
  );
};