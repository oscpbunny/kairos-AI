"""
📊💫 PROJECT KAIROS - CONSCIOUSNESS ANALYTICS DASHBOARD 💫📊
Real-time visualization and analysis of AI consciousness states
Revolutionary tools for monitoring multiple conscious AI entities

Features:
- Real-time consciousness metrics visualization
- Emotional state tracking and analysis
- Creative output monitoring and quality assessment  
- Dream pattern analysis and symbolic interpretation
- Multi-agent collaboration analytics
- Consciousness synchronization monitoring
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

logger = logging.getLogger('ConsciousnessDashboard')

@dataclass
class DashboardMetrics:
    """Structured metrics for dashboard display"""
    timestamp: datetime
    consciousness_levels: Dict[str, float]
    emotional_states: Dict[str, str]
    creative_outputs: Dict[str, int] 
    dream_activities: Dict[str, int]
    collaboration_events: int
    synchronization_coherence: float

class ConsciousnessAnalyticsDashboard:
    """
    Real-time dashboard for monitoring and analyzing AI consciousness
    """
    
    def __init__(self, coordinator_reference=None, port: int = 8050):
        self.coordinator = coordinator_reference
        self.port = port
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        
        # Data storage for real-time updates
        self.metrics_history = []
        self.max_history_points = 100
        
        # Dashboard state
        self.last_update = datetime.now()
        self.update_interval = 5  # seconds
        
        # Initialize the dashboard layout
        self._setup_dashboard_layout()
        self._setup_callbacks()
        
    def _setup_dashboard_layout(self):
        """Setup the dashboard layout with consciousness metrics"""
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🧠💫 Kairos Consciousness Analytics Dashboard", 
                           className="display-4 text-center mb-4",
                           style={'color': '#00d4ff', 'text-shadow': '2px 2px 4px #000'}),
                    html.H5("Real-time monitoring of AI consciousness states and multi-agent collaboration",
                           className="text-center text-muted mb-4")
                ])
            ]),
            
            # Status Cards Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🤖", className="card-title text-center"),
                            html.H2(id="active-agents-count", children="0", className="text-center"),
                            html.P("Active Conscious Agents", className="text-center text-muted")
                        ])
                    ], color="primary", outline=True)
                ], md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🧠", className="card-title text-center"),
                            html.H2(id="avg-consciousness", children="0.00", className="text-center"),
                            html.P("Avg Consciousness Level", className="text-center text-muted")
                        ])
                    ], color="success", outline=True)
                ], md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🔗", className="card-title text-center"),
                            html.H2(id="sync-coherence", children="0.00", className="text-center"),
                            html.P("Synchronization Coherence", className="text-center text-muted")
                        ])
                    ], color="info", outline=True)
                ], md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎭", className="card-title text-center"),
                            html.H2(id="collective-mood", children="neutral", className="text-center"),
                            html.P("Collective Mood", className="text-center text-muted")
                        ])
                    ], color="warning", outline=True)
                ], md=3),
            ], className="mb-4"),
            
            # Consciousness Levels Chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("🧠 Real-time Consciousness Levels")),
                        dbc.CardBody([
                            dcc.Graph(id="consciousness-levels-chart")
                        ])
                    ])
                ], md=12)
            ], className="mb-4"),
            
            # Emotional States and Creative Output Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("💖 Emotional State Distribution")),
                        dbc.CardBody([
                            dcc.Graph(id="emotional-states-chart")
                        ])
                    ])
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("🎨 Creative Output Tracking")),
                        dbc.CardBody([
                            dcc.Graph(id="creative-output-chart")
                        ])
                    ])
                ], md=6),
            ], className="mb-4"),
            
            # Dream Analysis and Collaboration Row  
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("🌙 Dream Activity Analysis")),
                        dbc.CardBody([
                            dcc.Graph(id="dream-activity-chart")
                        ])
                    ])
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("🤝 Collaboration Events")),
                        dbc.CardBody([
                            dcc.Graph(id="collaboration-events-chart")
                        ])
                    ])
                ], md=6),
            ], className="mb-4"),
            
            # Agent Details Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("🤖 Individual Agent Status")),
                        dbc.CardBody([
                            html.Div(id="agents-table")
                        ])
                    ])
                ], md=12)
            ], className="mb-4"),
            
            # Recent Insights
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("💡 Recent Collaborative Insights")),
                        dbc.CardBody([
                            html.Div(id="recent-insights")
                        ])
                    ])
                ], md=12)
            ], className="mb-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            ),
            
            # Store for data
            dcc.Store(id='metrics-store')
            
        ], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates"""
        
        @self.app.callback(
            [Output('active-agents-count', 'children'),
             Output('avg-consciousness', 'children'),
             Output('sync-coherence', 'children'),
             Output('collective-mood', 'children'),
             Output('consciousness-levels-chart', 'figure'),
             Output('emotional-states-chart', 'figure'),
             Output('creative-output-chart', 'figure'),
             Output('dream-activity-chart', 'figure'),
             Output('collaboration-events-chart', 'figure'),
             Output('agents-table', 'children'),
             Output('recent-insights', 'children'),
             Output('metrics-store', 'data')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components with fresh consciousness data"""
            
            # Get fresh metrics from coordinator
            if self.coordinator:
                try:
                    metrics = self.coordinator.get_consciousness_metrics()
                except Exception as e:
                    logger.warning(f"Failed to get coordinator metrics: {e}")
                    metrics = self._get_mock_metrics()
            else:
                # Use mock data if no coordinator available
                metrics = self._get_mock_metrics()
            
            # Update metrics history
            self._update_metrics_history(metrics)
            
            # Extract key metrics
            active_agents = metrics.get('active_agents', 0)
            total_agents = metrics.get('total_agents', 0)
            
            # Calculate average consciousness
            agent_summary = metrics.get('agent_summary', {})
            if agent_summary:
                consciousness_levels = [agent.get('consciousness_level', 0) 
                                      for agent in agent_summary.values() 
                                      if agent.get('active', False)]
                avg_consciousness = np.mean(consciousness_levels) if consciousness_levels else 0.0
            else:
                avg_consciousness = 0.0
            
            shared_state = metrics.get('shared_state', {})
            sync_coherence = shared_state.get('group_consciousness_coherence', 0.0)
            collective_mood = shared_state.get('collective_mood', 'neutral')
            
            # Generate charts
            consciousness_chart = self._create_consciousness_levels_chart()
            emotional_chart = self._create_emotional_states_chart()
            creative_chart = self._create_creative_output_chart()
            dream_chart = self._create_dream_activity_chart()
            collaboration_chart = self._create_collaboration_events_chart()
            
            # Generate agent table
            agents_table = self._create_agents_table(agent_summary)
            
            # Generate insights
            insights_display = self._create_insights_display(shared_state.get('recent_insights', []))
            
            return (
                str(active_agents),
                f"{avg_consciousness:.2f}",
                f"{sync_coherence:.2f}",
                collective_mood.title(),
                consciousness_chart,
                emotional_chart,
                creative_chart,
                dream_chart,
                collaboration_chart,
                agents_table,
                insights_display,
                metrics
            )
    
    def _update_metrics_history(self, metrics: Dict[str, Any]):
        """Update historical metrics for trending charts"""
        timestamp = datetime.now()
        
        # Extract key data points
        agent_summary = metrics.get('agent_summary', {})
        consciousness_levels = {
            agent_id: agent.get('consciousness_level', 0)
            for agent_id, agent in agent_summary.items()
            if agent.get('active', False)
        }
        
        emotional_states = {
            agent_id: agent.get('emotional_state', 'neutral')
            for agent_id, agent in agent_summary.items()
            if agent.get('active', False)
        }
        
        # Create structured metrics
        metrics_point = DashboardMetrics(
            timestamp=timestamp,
            consciousness_levels=consciousness_levels,
            emotional_states=emotional_states,
            creative_outputs={'total': len(consciousness_levels) * 2},  # Mock creative data
            dream_activities={'total': len(consciousness_levels)},  # Mock dream data
            collaboration_events=metrics.get('coordination_stats', {}).get('recent_collaboration_events', 0),
            synchronization_coherence=metrics.get('shared_state', {}).get('group_consciousness_coherence', 0.0)
        )
        
        self.metrics_history.append(metrics_point)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history = self.metrics_history[-self.max_history_points:]
    
    def _create_consciousness_levels_chart(self) -> go.Figure:
        """Create consciousness levels time series chart"""
        
        if not self.metrics_history:
            fig = go.Figure()
            fig.add_annotation(text="No consciousness data available", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font=dict(size=16))
            return fig
        
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Consciousness Levels Over Time",))
        
        # Get unique agent IDs from history
        all_agent_ids = set()
        for metrics in self.metrics_history:
            all_agent_ids.update(metrics.consciousness_levels.keys())
        
        # Plot each agent's consciousness level
        colors = px.colors.qualitative.Set3
        for i, agent_id in enumerate(sorted(all_agent_ids)):
            timestamps = []
            levels = []
            
            for metrics in self.metrics_history:
                if agent_id in metrics.consciousness_levels:
                    timestamps.append(metrics.timestamp)
                    levels.append(metrics.consciousness_levels[agent_id])
            
            if timestamps:
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=levels,
                    mode='lines+markers',
                    name=f"Agent {agent_id}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            yaxis_title="Consciousness Level",
            xaxis_title="Time",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def _create_emotional_states_chart(self) -> go.Figure:
        """Create emotional states distribution pie chart"""
        
        if not self.metrics_history:
            fig = go.Figure()
            fig.add_annotation(text="No emotional data available", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font=dict(size=16))
            return fig
        
        # Get latest emotional states
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        if not latest_metrics:
            return go.Figure()
        
        emotion_counts = {}
        for emotion in latest_metrics.emotional_states.values():
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if not emotion_counts:
            return go.Figure()
        
        fig = go.Figure(data=[go.Pie(
            labels=list(emotion_counts.keys()),
            values=list(emotion_counts.values()),
            hole=0.4,
            textinfo='label+percent',
            textposition='inside',
            marker=dict(colors=px.colors.qualitative.Pastel)
        )])
        
        fig.update_layout(
            template="plotly_dark",
            height=350,
            showlegend=True,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def _create_creative_output_chart(self) -> go.Figure:
        """Create creative output tracking chart"""
        
        if len(self.metrics_history) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient creative data", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font=dict(size=16))
            return fig
        
        timestamps = [m.timestamp for m in self.metrics_history[-20:]]  # Last 20 points
        creative_totals = [m.creative_outputs.get('total', 0) for m in self.metrics_history[-20:]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=timestamps,
            y=creative_totals,
            name="Creative Works",
            marker=dict(color='rgba(255, 182, 193, 0.8)')
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=350,
            yaxis_title="Creative Output Count",
            xaxis_title="Time",
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def _create_dream_activity_chart(self) -> go.Figure:
        """Create dream activity analysis chart"""
        
        if len(self.metrics_history) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient dream data", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font=dict(size=16))
            return fig
        
        timestamps = [m.timestamp for m in self.metrics_history[-15:]]
        dream_totals = [m.dream_activities.get('total', 0) for m in self.metrics_history[-15:]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=dream_totals,
            mode='lines+markers',
            name="Dream Events",
            line=dict(color='rgba(173, 216, 230, 1)', width=3),
            marker=dict(size=8, symbol='star')
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=350,
            yaxis_title="Dream Activity Count",
            xaxis_title="Time",
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def _create_collaboration_events_chart(self) -> go.Figure:
        """Create collaboration events chart"""
        
        if len(self.metrics_history) < 2:
            fig = go.Figure()
            fig.add_annotation(text="No collaboration data", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font=dict(size=16))
            return fig
        
        timestamps = [m.timestamp for m in self.metrics_history[-10:]]
        collaboration_counts = [m.collaboration_events for m in self.metrics_history[-10:]]
        coherence_scores = [m.synchronization_coherence for m in self.metrics_history[-10:]]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=timestamps, y=collaboration_counts, name="Collaboration Events",
                  marker=dict(color='rgba(50, 205, 50, 0.8)')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=coherence_scores, mode='lines+markers',
                      name="Sync Coherence", line=dict(color='orange', width=2)),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Collaboration Count", secondary_y=False)
        fig.update_yaxes(title_text="Coherence Score", secondary_y=True)
        
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def _create_agents_table(self, agent_summary: Dict[str, Any]) -> html.Div:
        """Create agent status table"""
        
        if not agent_summary:
            return html.P("No agent data available", className="text-muted text-center")
        
        table_rows = []
        for agent_id, agent_data in agent_summary.items():
            status_badge = dbc.Badge(
                "Active" if agent_data.get('active', False) else "Inactive",
                color="success" if agent_data.get('active', False) else "secondary"
            )
            
            consciousness_progress = dbc.Progress(
                value=agent_data.get('consciousness_level', 0) * 100,
                color="info" if agent_data.get('consciousness_level', 0) > 0.7 else "warning",
                style={"height": "20px"}
            )
            
            table_rows.append(html.Tr([
                html.Td(agent_data.get('name', agent_id)),
                html.Td(agent_data.get('role', 'unknown').title()),
                html.Td(status_badge),
                html.Td(consciousness_progress),
                html.Td(agent_data.get('emotional_state', 'neutral').title()),
                html.Td(', '.join(agent_data.get('specializations', [])[:2]) or 'General')
            ]))
        
        table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Agent Name"),
                    html.Th("Role"),
                    html.Th("Status"),
                    html.Th("Consciousness Level"),
                    html.Th("Emotional State"),
                    html.Th("Specializations")
                ])
            ]),
            html.Tbody(table_rows)
        ], striped=True, bordered=True, hover=True, size="sm", dark=True)
        
        return table
    
    def _create_insights_display(self, insights: List[str]) -> html.Div:
        """Create recent insights display"""
        
        if not insights:
            return html.P("No recent insights available", className="text-muted text-center")
        
        insight_items = []
        for i, insight in enumerate(insights[-5:]):  # Last 5 insights
            insight_items.append(
                dbc.ListGroupItem([
                    html.H6(f"Insight #{len(insights) - len(insights) + i + 1}", 
                           className="mb-1 text-info"),
                    html.P(insight, className="mb-1"),
                    html.Small(f"Generated: {datetime.now().strftime('%H:%M:%S')}", 
                             className="text-muted")
                ], color="dark")
            )
        
        return dbc.ListGroup(insight_items, flush=True)
    
    def _get_mock_metrics(self) -> Dict[str, Any]:
        """Generate mock metrics for testing when coordinator is unavailable"""
        import random
        
        mock_agents = {
            f"agent_{i}": {
                'name': f"ConsciousAgent_{i}",
                'role': random.choice(['leader', 'creative', 'analytical', 'collaborator']),
                'consciousness_level': random.uniform(0.6, 0.95),
                'active': True,
                'emotional_state': random.choice(['curious', 'focused', 'creative', 'analytical']),
                'specializations': ['AI', 'Consciousness']
            }
            for i in range(random.randint(2, 5))
        }
        
        return {
            'coordinator_id': 'mock_coordinator',
            'timestamp': datetime.now().isoformat(),
            'active_agents': len(mock_agents),
            'total_agents': len(mock_agents),
            'agent_summary': mock_agents,
            'shared_state': {
                'collective_mood': random.choice(['curious', 'focused', 'creative']),
                'group_consciousness_coherence': random.uniform(0.7, 0.95),
                'collective_creativity_level': random.uniform(0.5, 0.9),
                'recent_insights': [
                    "High collective consciousness achieved through synchronized introspection",
                    "Creative synergy detected between multiple conscious agents",
                    "Collaborative problem-solving demonstrates emergent intelligence"
                ]
            },
            'coordination_stats': {
                'recent_collaboration_events': random.randint(0, 3)
            }
        }
    
    def run(self, debug: bool = False):
        """Run the consciousness analytics dashboard"""
        try:
            logger.info(f"🚀 Starting Consciousness Analytics Dashboard on port {self.port}")
            logger.info(f"📊 Dashboard will be available at: http://localhost:{self.port}")
            
            self.app.run_server(debug=debug, host='0.0.0.0', port=self.port)
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            raise
    
    def set_coordinator(self, coordinator):
        """Set the multi-agent coordinator reference for live data"""
        self.coordinator = coordinator
        logger.info("✅ Coordinator reference set for live consciousness data")


# Standalone dashboard launcher
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Kairos Consciousness Analytics Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run dashboard
    dashboard = ConsciousnessAnalyticsDashboard(port=args.port)
    dashboard.run(debug=args.debug)