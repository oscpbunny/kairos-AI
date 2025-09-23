#!/usr/bin/env python3
"""
Kairos Vision Board - A Real-Time TUI Dashboard
Mission control for the Kairos Autonomous Digital Organization.

Author: Kairos Development Team
Version: 1.0
"""

import asyncio
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import time
import logging

from .api_client import KairosAPIClient
from .components import (
    create_header,
    create_system_status_panel,
    create_agent_swarm_panel,
    create_causal_ledger_panel,
    create_economic_dashboard_panel,
    create_oracle_insights_panel,
    create_venture_progress_panel,
    create_footer
)


class VisionBoard:
    """
    The main class for the TUI dashboard.
    """
    def __init__(self):
        self.console = Console()
        self.layout = self.make_layout()
        self.api_client = KairosAPIClient()
        self.dashboard_data = {
            "system_health": {},
            "agents": [],
            "decisions": [],
            "economic_metrics": {},
            "oracle_insights": [],
            "ventures": []
        }

    def make_layout(self) -> Layout:
        """Define the dashboard layout."""
        layout = Layout(name="root")

        layout.split(
            Layout(name="header", size=3),
            Layout(ratio=1, name="main"),
            Layout(size=1, name="footer"),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right", ratio=2),
        )

        layout["left"].split(
            Layout(name="system_status"),
            Layout(name="economy"),
            Layout(name="oracle")
        )
        
        layout["right"].split(
            Layout(name="agent_swarm"),
            Layout(name="causal_ledger"),
            Layout(name="venture_progress")
        )
        return layout

    async def update_data(self):
        """Coroutine to fetch snapshot data periodically."""
        while True:
            data = await self.api_client.fetch_all_data()
            if data:
                self.dashboard_data.update(data)
            await asyncio.sleep(5) # Update every 5 seconds

    async def listen_for_decisions(self):
        """Listen for real-time decision updates."""
        query = "subscription { decision_stream { id agent_id decision_type timestamp } }"
        async def decision_callback(data):
            self.dashboard_data['decisions'].insert(0, data['decision_stream'])
            if len(self.dashboard_data['decisions']) > 10:
                self.dashboard_data['decisions'].pop()
        await self.api_client.subscribe(query, decision_callback)

    async def listen_for_agent_updates(self):
        """Listen for real-time agent activity."""
        query = "subscription { agent_activity { id name role status cc_balance reputation_score } }"
        async def agent_callback(data):
            updated_agent = data['agent_activity']
            for i, agent in enumerate(self.dashboard_data['agents']):
                if agent['id'] == updated_agent['id']:
                    self.dashboard_data['agents'][i] = updated_agent
                    break
            else:
                self.dashboard_data['agents'].append(updated_agent)
        await self.api_client.subscribe(query, agent_callback)


    def update_layout(self):
        """Update the layout panels with the latest data."""
        self.layout["header"].update(create_header())
        self.layout["system_status"].update(create_system_status_panel(self.dashboard_data))
        self.layout["agent_swarm"].update(create_agent_swarm_panel(self.dashboard_data))
        self.layout["causal_ledger"].update(create_causal_ledger_panel(self.dashboard_data))
        self.layout["economy"].update(create_economic_dashboard_panel(self.dashboard_data))
        self.layout["oracle"].update(create_oracle_insights_panel(self.dashboard_data))
        self.layout["venture_progress"].update(create_venture_progress_panel(self.dashboard_data))
        self.layout["footer"].update(create_footer())

    async def run(self):
        """
        Run the dashboard, updating it in real-time.
        """
        tasks = [
            asyncio.create_task(self.update_data()),
            asyncio.create_task(self.listen_for_decisions()),
            asyncio.create_task(self.listen_for_agent_updates())
        ]

        with Live(self.layout, screen=True, redirect_stderr=False) as live:
            try:
                while True:
                    self.update_layout()
                    await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                for task in tasks:
                    task.cancel()


def main():
    # Set up logging
    logging.basicConfig(filename='E:\\kairos\\logs\\vision_board.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Initializing Kairos Vision Board")

    print("""
    Kairos Vision Board - Real-Time Dashboard
    -------------------------------------------
    Instructions:
    1. Make sure the Kairos API server is running.
       (You can run it with: python api/launcher.py)
    2. This dashboard will connect to the API to display live data.
    3. Press CTRL+C to exit the dashboard.
    """)
    time.sleep(3)

    board = VisionBoard()
    try:
        asyncio.run(board.run())
    except Exception as e:
        logger.error(f"Failed to run Vision Board: {e}")
        print(f"An error occurred: {e}")
        print(f"Check the log file at E:\\kairos\\logs\\vision_board.log")

if __name__ == "__main__":
    main()
