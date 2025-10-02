"use client";

import React from "react";

export const LogsPanel: React.FC<{ className?: string }> = ({ className = "" }) => {
  const logs = [
    { level: 'info', msg: 'System started', ts: new Date().toLocaleTimeString() },
    { level: 'info', msg: 'Agents initialized', ts: new Date().toLocaleTimeString() },
    { level: 'warn', msg: 'Low memory warning (mock)', ts: new Date().toLocaleTimeString() },
  ];
  const color = (lvl: string) => lvl === 'warn' ? 'text-yellow-600' : lvl === 'error' ? 'text-red-600' : 'text-gray-600';
  return (
    <div className={`p-6 space-y-4 ${className}`}>
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Recent Logs</h3>
        <div className="space-y-2 text-sm">
          {logs.map((l, i) => (
            <div key={i} className="flex items-center justify-between">
              <span className={color(l.level)}>[{l.level.toUpperCase()}]</span>
              <span className="flex-1 mx-3 truncate">{l.msg}</span>
              <span className="text-xs text-gray-500">{l.ts}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
