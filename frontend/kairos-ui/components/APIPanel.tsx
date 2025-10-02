"use client";

import React from "react";

export const APIPanel: React.FC<{ className?: string }> = ({ className = "" }) => {
  return (
    <div className={`p-6 space-y-4 ${className}`}>
      <div className="card p-4">
        <h3 className="font-semibold mb-3">API Explorer</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">Document and test Kairos API endpoints (mock UI).</p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
          {['GET /agents', 'POST /chat/send', 'GET /system/metrics'].map((ep) => (
            <div key={ep} className="card p-3">
              <div className="text-sm font-medium">{ep}</div>
              <button className="btn btn-secondary mt-2 text-xs">Try it</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
