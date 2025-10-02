"use client";

import React from "react";

export const DockerPanel: React.FC<{ className?: string }> = ({ className = "" }) => {
  return (
    <div className={`p-6 space-y-4 ${className}`}>
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Docker</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">Manage containers (mock UI).</p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
          {['kairos-backend', 'kairos-worker'].map((c) => (
            <div key={c} className="card p-3 flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">{c}</div>
                <div className="text-xs text-gray-500">Running</div>
              </div>
              <div className="space-x-2">
                <button className="btn btn-secondary text-xs">Restart</button>
                <button className="btn btn-secondary text-xs">Stop</button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
