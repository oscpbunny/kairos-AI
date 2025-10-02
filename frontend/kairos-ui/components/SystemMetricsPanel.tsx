"use client";

import React from "react";

export const SystemMetricsPanel: React.FC<{ className?: string }> = ({ className = "" }) => {
  return (
    <div className={`p-6 space-y-4 ${className}`}>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[{label:'Agents Online', value: 4, color:'text-green-600'}, {label:'Tasks Active', value: 7, color:'text-blue-600'}, {label:'Avg Latency', value: '1.2s', color:'text-yellow-600'}, {label:'Uptime', value: '99.97%', color:'text-emerald-600'}].map((m) => (
          <div key={m.label} className="card p-4">
            <div className="text-sm text-gray-500">{m.label}</div>
            <div className={`text-2xl font-bold ${m.color}`}>{m.value}</div>
          </div>
        ))}
      </div>

      <div className="card p-6">
        <h3 className="font-semibold mb-4">System Overview</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          All systems nominal. Agents collaborating effectively. No incidents detected.
        </p>
      </div>
    </div>
  );
};
