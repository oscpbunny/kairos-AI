"use client";

import React from "react";

export const SettingsPanel: React.FC<{ className?: string }> = ({ className = "" }) => {
  return (
    <div className={`p-6 space-y-4 ${className}`}>
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Preferences</h3>
        <div className="space-y-3 text-sm">
          <label className="flex items-center justify-between">
            <span>Dark mode</span>
            <input type="checkbox" className="toggle" defaultChecked />
          </label>
          <label className="flex items-center justify-between">
            <span>Agent notifications</span>
            <input type="checkbox" className="toggle" defaultChecked />
          </label>
          <label className="flex items-center justify-between">
            <span>Auto-start tasks</span>
            <input type="checkbox" className="toggle" />
          </label>
        </div>
      </div>
    </div>
  );
};
