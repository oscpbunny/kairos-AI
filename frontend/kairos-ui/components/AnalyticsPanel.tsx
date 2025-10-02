"use client";

import React from "react";
import { AnalyticsDashboard } from "./AnalyticsDashboard";

export const AnalyticsPanel: React.FC<{ className?: string }> = ({ className = "" }) => {
  return <AnalyticsDashboard className={className} /> as any;
};
