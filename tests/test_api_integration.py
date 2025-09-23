"""
Project Kairos: API Integration Tests
Tests for both GraphQL and gRPC endpoints.
"""

import asyncio
import pytest
import httpx
import json
import time
from typing import Dict, Any

# GraphQL Test Suite
class TestGraphQLAPI:
    """Test GraphQL API endpoints"""
    
    BASE_URL = "http://localhost:8000"
    GRAPHQL_ENDPOINT = f"{BASE_URL}/graphql"
    
    async def test_graphql_health_query(self):
        """Test basic GraphQL health query"""
        query = """
        query {
            hello
        }
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": query}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert data["data"]["hello"] == "Hello from Project Kairos GraphQL API!"
    
    async def test_system_health_query(self):
        """Test system health GraphQL query"""
        query = """
        query {
            system_health {
                overall_status
                healthy_components
                total_components
                success_rate
                components {
                    component
                    status
                    healthy
                }
            }
        }
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": query}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "system_health" in data["data"]
            
            health = data["data"]["system_health"]
            assert health["overall_status"] in ["HEALTHY", "DEGRADED", "UNHEALTHY"]
            assert isinstance(health["healthy_components"], int)
            assert isinstance(health["total_components"], int)
            assert isinstance(health["success_rate"], float)
            assert isinstance(health["components"], list)
    
    async def test_agents_query(self):
        """Test agents GraphQL query"""
        query = """
        query {
            agents {
                id
                name
                agent_type
                specialization
                cognitive_cycles_balance
                is_active
            }
        }
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": query}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "agents" in data["data"]
            
            agents = data["data"]["agents"]
            assert isinstance(agents, list)
            
            if agents:  # If we have mock agents
                agent = agents[0]
                assert "id" in agent
                assert "name" in agent
                assert "agent_type" in agent
                assert agent["agent_type"] in ["STEWARD", "ARCHITECT", "ENGINEER", "ORACLE"]
    
    async def test_ventures_query(self):
        """Test ventures GraphQL query"""
        query = """
        query {
            ventures {
                id
                name
                objective
                status
                target_users
                budget
            }
        }
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": query}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "ventures" in data["data"]
            
            ventures = data["data"]["ventures"]
            assert isinstance(ventures, list)
    
    async def test_create_venture_mutation(self):
        """Test create venture GraphQL mutation"""
        mutation = """
        mutation {
            create_venture(input: {
                name: "Test API Venture"
                objective: "Test the GraphQL API functionality"
                target_users: 1000
                budget: 5000.0
                timeline_days: 30
            }) {
                id
                name
                objective
                status
                target_users
                budget
            }
        }
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": mutation}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "create_venture" in data["data"]
            
            venture = data["data"]["create_venture"]
            assert venture["name"] == "Test API Venture"
            assert venture["objective"] == "Test the GraphQL API functionality"
            assert venture["status"] == "PLANNING"
            assert venture["target_users"] == 1000
            assert venture["budget"] == 5000.0
    
    async def test_infrastructure_prediction_query(self):
        """Test infrastructure prediction GraphQL query"""
        query = """
        query {
            predict_infrastructure(input: {
                venture_id: "venture-001"
                time_horizon_days: 30
                current_infrastructure: "{}"
            }) {
                venture_id
                time_horizon_days
                monthly_estimate
                confidence
                recommendations
            }
        }
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": query}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "predict_infrastructure" in data["data"]
            
            prediction = data["data"]["predict_infrastructure"]
            assert prediction["venture_id"] == "venture-001"
            assert prediction["time_horizon_days"] == 30
            assert isinstance(prediction["monthly_estimate"], (int, float))
            assert isinstance(prediction["confidence"], float)
            assert isinstance(prediction["recommendations"], list)

class TestRESTAPI:
    """Test REST API endpoints"""
    
    BASE_URL = "http://localhost:8000"
    
    async def test_root_endpoint(self):
        """Test root endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(self.BASE_URL)
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert data["message"] == "Welcome to Project Kairos API"
    
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/health")
            
            assert response.status_code in [200, 206, 503]  # Healthy, degraded, or unhealthy
            data = response.json()
            assert "status" in data
            assert data["status"] in ["healthy", "degraded", "unhealthy"]
            assert "components" in data
    
    async def test_login_endpoint(self):
        """Test authentication endpoint"""
        login_data = {
            "username": "admin",
            "password": "kairos-admin"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/auth/login",
                json=login_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"
            assert "scopes" in data
            
            return data["access_token"]  # Return token for other tests
    
    async def test_protected_ventures_endpoint(self):
        """Test protected ventures endpoint"""
        # First get auth token
        token = await self.test_login_endpoint()
        
        headers = {"Authorization": f"Bearer {token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/api/v1/ventures",
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
    
    async def test_protected_agents_endpoint(self):
        """Test protected agents endpoint"""
        # First get auth token
        token = await self.test_login_endpoint()
        
        headers = {"Authorization": f"Bearer {token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/api/v1/agents",
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
    
    async def test_infrastructure_prediction_endpoint(self):
        """Test infrastructure prediction REST endpoint"""
        # First get auth token
        token = await self.test_login_endpoint()
        
        headers = {"Authorization": f"Bearer {token}"}
        prediction_data = {
            "venture_id": "venture-001",
            "time_horizon_days": 30,
            "current_infrastructure": "{}"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/api/v1/infrastructure/predict",
                headers=headers,
                json=prediction_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "venture_id" in data
            assert "monthly_estimate" in data
            assert "confidence" in data

class TestgRPCAPI:
    """Test gRPC API endpoints"""
    
    async def test_grpc_connection(self):
        """Test basic gRPC connection"""
        try:
            import grpc
            
            # Create channel
            channel = grpc.aio.insecure_channel('localhost:50051')
            
            # Test connection state
            await channel.channel_ready()
            state = channel.get_state()
            
            assert state == grpc.ChannelConnectivity.READY
            
            await channel.close()
            
        except ImportError:
            pytest.skip("gRPC not available")
        except Exception as e:
            pytest.fail(f"gRPC connection test failed: {e}")

# Test Runner Functions
async def run_graphql_tests():
    """Run GraphQL tests"""
    print("🧪 Running GraphQL Tests...")
    
    test_instance = TestGraphQLAPI()
    
    tests = [
        test_instance.test_graphql_health_query,
        test_instance.test_system_health_query,
        test_instance.test_agents_query,
        test_instance.test_ventures_query,
        test_instance.test_create_venture_mutation,
        test_instance.test_infrastructure_prediction_query,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print(f"\nGraphQL Tests: {passed} passed, {failed} failed")
    return passed, failed

async def run_rest_tests():
    """Run REST API tests"""
    print("\n🧪 Running REST API Tests...")
    
    test_instance = TestRESTAPI()
    
    tests = [
        test_instance.test_root_endpoint,
        test_instance.test_health_endpoint,
        test_instance.test_login_endpoint,
        test_instance.test_protected_ventures_endpoint,
        test_instance.test_protected_agents_endpoint,
        test_instance.test_infrastructure_prediction_endpoint,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print(f"\nREST API Tests: {passed} passed, {failed} failed")
    return passed, failed

async def run_grpc_tests():
    """Run gRPC tests"""
    print("\n🧪 Running gRPC Tests...")
    
    test_instance = TestgRPCAPI()
    
    tests = [
        test_instance.test_grpc_connection,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print(f"\ngRPC Tests: {passed} passed, {failed} failed")
    return passed, failed

async def run_all_tests():
    """Run all API integration tests"""
    print("🚀 Project Kairos API Integration Tests")
    print("=" * 50)
    
    # Wait for servers to be ready
    print("⏳ Waiting for servers to be ready...")
    await asyncio.sleep(2)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    graphql_passed, graphql_failed = await run_graphql_tests()
    rest_passed, rest_failed = await run_rest_tests()
    grpc_passed, grpc_failed = await run_grpc_tests()
    
    total_passed = graphql_passed + rest_passed + grpc_passed
    total_failed = graphql_failed + rest_failed + grpc_failed
    
    print(f"\n{'=' * 50}")
    print(f"📊 Final Results:")
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"📈 Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%" if (total_passed + total_failed) > 0 else "N/A")
    
    return total_passed, total_failed

if __name__ == "__main__":
    # Run tests
    passed, failed = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    exit_code = 0 if failed == 0 else 1
    exit(exit_code)