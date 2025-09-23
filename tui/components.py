#!/usr/bin/env python3
"""
UI Components for the Kairos Vision Board
"""

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from datetime import datetime

def create_header() -> Panel:
    """Creates the header panel with the title."""
    return Panel(Text("Project Kairos - Vision Board", justify="center", style="bold magenta"), border_style="magenta")

def create_system_status_panel(data: dict) -> Panel:
    """Creates the system status panel."""
    health = data.get('system_health', {})
    status = "HEALTHY" if health.get('healthy') else "UNHEALTHY"
    active_agents = health.get('active_agents', 0)
    pending_tasks = health.get('pending_tasks', 0)

    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row("Overall Status:", Text(status, style="green" if status == "HEALTHY" else "red"))
    grid.add_row("Active Agents:", str(active_agents))
    grid.add_row("Pending Tasks:", str(pending_tasks))
    return Panel(grid, title="[bold]System Health[/bold]", border_style="green")

def create_agent_swarm_panel(data: dict) -> Panel:
    """Creates the agent swarm status panel."""
    agents = data.get('agents', [])
    table = Table(title="Agent Swarm Status", expand=True)

    table.add_column("ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Role", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("CC Balance", justify="right", style="green")
    table.add_column("Reputation", justify="right", style="blue")

    for agent in agents:
        status_style = "green" if agent['status'] == "ACTIVE" else "yellow"
        table.add_row(
            agent['id'],
            agent['role'],
            Text(agent['status'], style=status_style),
            f"{agent['cc_balance']:.2f}",
            f"{agent['reputation_score']:.3f}"
        )
    return Panel(table, border_style="cyan")

def create_causal_ledger_panel(data: dict) -> Panel:
    """Creates the live causal ledger panel."""
    decisions = data.get('decisions', [])
    text = Text()
    for decision in decisions:
        text.append(f"{decision['timestamp']} - {decision['agent_id']} - {decision['decision_type']}\n")
    return Panel(text, title="[bold]Live Causal Ledger[/bold]", border_style="yellow")

def create_economic_dashboard_panel(data: dict) -> Panel:
    """Creates the internal economy dashboard panel."""
    economy = data.get('economic_metrics', {})
    total_cc = economy.get('total_cc_circulation', 0)
    efficiency = economy.get('market_efficiency_score', 0)
    
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row("Total CC Circulation:", f"{total_cc:.2f}")
    grid.add_row("Market Efficiency:", f"{efficiency:.2%}")
    return Panel(grid, title="[bold]Internal Economy[/bold]", border_style="blue")

def create_oracle_insights_panel(data: dict) -> Panel:
    """Creates the Oracle insights panel."""
    insights = data.get('oracle_insights', [])
    table = Table(title="Oracle Insights", expand=True)
    table.add_column("ID", style="dim")
    table.add_column("Scenario", style="yellow")
    table.add_column("Confidence", style="green")

    for insight in insights:
        table.add_row(
            insight['prediction_id'][:8],
            insight['scenario_type'],
            f"{insight['confidence_score']:.2%}"
        )
    return Panel(table, border_style="red")

def create_venture_progress_panel(data: dict) -> Panel:
    """Creates the venture progress panel."""
    ventures = data.get('ventures', [])
    table = Table(title="Venture Progress", expand=True)
    table.add_column("ID", style="dim")
    table.add_column("Name", style="purple")
    table.add_column("Status", style="green")
    table.add_column("Completion", justify="right", style="blue")

    for venture in ventures:
        table.add_row(
            venture['id'][:8],
            venture['name'],
            venture['status'],
            f"{venture.get('completion_percentage', 0):.1%}"
        )
    return Panel(table, border_style="purple")

def create_footer() -> Text:
    """Creates the footer text."""
    return Text("Press CTRL+C to exit | All data is live from the Kairos API", justify="center", style="dim")
