#!/usr/bin/env python3
"""
Kairos API Client - for the Vision Board
Handles all communication with the Kairos GraphQL API.
"""

import asyncio
import json
import logging
import httpx
import websockets
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

class KairosAPIClient:
    """
    A client to interact with the Kairos GraphQL API.
    """
    def __init__(self, http_url="http://localhost:8000/graphql", ws_url="ws://localhost:8000/graphql"):
        self.http_url = http_url
        self.ws_url = ws_url
        self.client = httpx.AsyncClient()

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a single GraphQL query."""
        try:
            response = await self.client.post(self.http_url, json={'query': query}, headers={"Authorization": "Bearer kairos_api_token"})
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}

    async def subscribe(self, query: str, callback: Callable):
        """Create a GraphQL subscription and stream results."""
        try:
            async with websockets.connect(self.ws_url, subprotocols=["graphql-ws"]) as websocket:
                # Initialize the connection
                await websocket.send(json.dumps({'type': 'connection_init'}))
                init_ack = await websocket.recv()
                if json.loads(init_ack)['type'] != 'connection_ack':
                    logger.error("Subscription connection not acknowledged.")
                    return

                # Start the subscription
                sub_id = "1"
                await websocket.send(json.dumps({
                    'id': sub_id,
                    'type': 'start',
                    'payload': {'query': query}
                }))

                # Listen for data
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data['type'] == 'data':
                        await callback(data['payload']['data'])
                    elif data['type'] == 'error':
                        logger.error(f"Subscription error: {data['payload']}")
                        break
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidURIError) as e:
            logger.error(f"Subscription connection failed: {e}")
            await asyncio.sleep(5) # Wait before retrying

    async def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch a snapshot of all necessary data for the dashboard."""
        query = """
        {
          system_health {
            healthy
            active_agents
            pending_tasks
          }
          agents(limit: 10) {
            id
            name
            role
            status
            cc_balance
            reputation_score
          }
          economic_metrics {
            total_cc_circulation
            market_efficiency_score
          }
          oracle_insights(limit: 5) {
            prediction_id
            scenario_type
            confidence_score
          }
          ventures(limit: 5) {
            id
            name
            status
            completion_percentage
          }
        }
        """
        response = await self.execute_query(query)
        if 'data' in response:
            return response['data']
        return {}
