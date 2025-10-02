"use client";

import React from "react";

export const DatabasePanel: React.FC<{ className?: string }> = ({ className = "" }) => {
  return (
    <div className={`p-6 space-y-4 ${className}`}>
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Database</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">Connect to your Kairos datastore (mock UI).</p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
          {["Collections", "Queries", "Backups"].map((t) => (
            <div key={t} className="card p-3">
              <div className="text-sm font-medium">{t}</div>
              <div className="text-xs text-gray-500">No items</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
