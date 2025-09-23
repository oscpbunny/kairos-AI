"""
🧠📊 KAIROS ML ANALYTICS DEMO 📊🧠
Demonstration of Advanced Machine Learning Analytics for Multi-Agent AI Coordination

Features:
- Performance clustering and segmentation
- Anomaly detection in agent behavior  
- Predictive analytics for system performance
- Pattern recognition in collaboration networks
- Automated insight generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class SimplifiedMLAnalytics:
    """Simplified ML Analytics for demonstration"""
    
    def __init__(self):
        self.system_data = []
        self.agent_data = []
        self.event_data = []
        
    def add_system_metrics(self, metrics_list):
        """Add system metrics for analysis"""
        self.system_data.extend(metrics_list)
        
    def add_agent_data(self, agents_list):
        """Add agent data for analysis"""
        self.agent_data.extend(agents_list)
        
    def add_collaboration_events(self, events_list):
        """Add collaboration events for analysis"""
        self.event_data.extend(events_list)
    
    def analyze_performance_clusters(self):
        """Analyze agent performance clusters"""
        print("🎯 Analyzing Agent Performance Clusters...")
        
        if len(self.agent_data) < 3:
            return {"message": "Insufficient agent data for clustering"}
            
        # Convert to DataFrame
        df = pd.DataFrame(self.agent_data)
        latest_agents = df.groupby('agent_id').last().reset_index()
        
        # Extract features for clustering
        features = latest_agents[['performance_score', 'tasks_completed', 'collaboration_count']].values
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform clustering
        n_clusters = min(3, len(latest_agents))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_agents = latest_agents[cluster_mask]
            
            avg_performance = cluster_agents['performance_score'].mean()
            if avg_performance >= 0.8:
                performance_level = "High Performance"
            elif avg_performance >= 0.6:
                performance_level = "Medium Performance"
            else:
                performance_level = "Low Performance"
                
            clusters[f"Cluster {cluster_id}"] = {
                "agents": cluster_agents['agent_id'].tolist(),
                "performance_level": performance_level,
                "avg_performance": avg_performance,
                "avg_tasks": cluster_agents['tasks_completed'].mean(),
                "avg_collaboration": cluster_agents['collaboration_count'].mean()
            }
        
        return {
            "clusters": clusters,
            "total_agents": len(latest_agents),
            "insights": [
                f"Identified {len(clusters)} distinct performance clusters",
                f"Analyzed {len(latest_agents)} active agents"
            ]
        }
    
    def detect_anomalies(self):
        """Detect system performance anomalies"""
        print("🚨 Detecting System Anomalies...")
        
        if len(self.system_data) < 10:
            return {"message": "Insufficient system data for anomaly detection"}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.system_data)
        
        # Extract features
        features = df[['coordination_quality', 'sync_performance', 'system_health', 'performance_score']].values
        features = np.nan_to_num(features)  # Handle NaN values
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features)
        
        # Count anomalies
        anomalies = np.sum(anomaly_labels == -1)
        anomaly_percentage = (anomalies / len(features)) * 100
        
        return {
            "anomalies_detected": int(anomalies),
            "total_data_points": len(features),
            "anomaly_percentage": anomaly_percentage,
            "insights": [
                f"Detected {anomalies} anomalies out of {len(features)} data points",
                f"Anomaly rate: {anomaly_percentage:.1f}%",
                "System operating normally" if anomaly_percentage < 5 else "High anomaly rate detected"
            ]
        }
    
    def predict_performance(self, horizon_minutes=30):
        """Predict future system performance"""
        print("🔮 Predicting System Performance...")
        
        if len(self.system_data) < 15:
            return {"message": "Insufficient data for prediction"}
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(self.system_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create time-based features
        df['time_numeric'] = df['timestamp'].astype(np.int64) // 10**9
        
        predictions = {}
        metrics = ['coordination_quality', 'sync_performance', 'system_health', 'performance_score']
        
        for metric in metrics:
            try:
                X = df[['time_numeric']].values
                y = df[metric].values
                
                # Handle NaN values
                mask = ~np.isnan(y)
                X, y = X[mask], y[mask]
                
                if len(y) < 5:
                    continue
                
                # Train model
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate future predictions
                last_time = df['time_numeric'].iloc[-1]
                future_times = []
                for i in range(1, horizon_minutes + 1):
                    future_time = last_time + (i * 60)
                    future_times.append([future_time])
                
                future_predictions = model.predict(future_times)
                
                # Calculate trend
                trend = "increasing" if future_predictions[-1] > future_predictions[0] else "decreasing"
                
                predictions[metric] = {
                    "trend": trend,
                    "predicted_values": future_predictions[:5].tolist(),  # First 5 minutes
                    "accuracy_score": model.score(X, y)
                }
                
            except Exception as e:
                continue
        
        return {
            "predictions": predictions,
            "horizon_minutes": horizon_minutes,
            "models_trained": len(predictions),
            "insights": [
                f"Generated predictions for {len(predictions)} metrics",
                f"Prediction horizon: {horizon_minutes} minutes"
            ]
        }
    
    def analyze_collaboration_network(self):
        """Analyze collaboration patterns"""
        print("🕸️ Analyzing Collaboration Network...")
        
        if len(self.event_data) < 5:
            return {"message": "Insufficient collaboration data"}
        
        # Build collaboration matrix
        from collections import defaultdict
        collaboration_matrix = defaultdict(lambda: defaultdict(int))
        agent_participation = defaultdict(int)
        success_rates = defaultdict(list)
        
        for event in self.event_data:
            participants = event.get('participants', [])
            success = event.get('success', True)
            
            # Count participation and success rates
            for agent in participants:
                agent_participation[agent] += 1
                success_rates[agent].append(success)
            
            # Build collaboration pairs
            for i, agent1 in enumerate(participants):
                for agent2 in participants[i+1:]:
                    collaboration_matrix[agent1][agent2] += 1
                    collaboration_matrix[agent2][agent1] += 1
        
        # Calculate metrics
        total_agents = len(agent_participation)
        total_collaborations = sum(len(collabs) for collabs in collaboration_matrix.values()) // 2
        
        # Network density
        max_possible = total_agents * (total_agents - 1) // 2
        density = total_collaborations / max_possible if max_possible > 0 else 0
        
        # Success rates
        agent_success_rates = {}
        for agent, successes in success_rates.items():
            agent_success_rates[agent] = sum(successes) / len(successes) if successes else 0
        
        avg_success_rate = np.mean(list(agent_success_rates.values())) if agent_success_rates else 0
        
        # Most collaborative agent
        most_collaborative = max(agent_participation.items(), key=lambda x: x[1]) if agent_participation else ("None", 0)
        
        return {
            "network_density": density,
            "total_agents": total_agents,
            "total_collaborations": total_collaborations,
            "average_success_rate": avg_success_rate,
            "most_collaborative_agent": most_collaborative[0],
            "agent_success_rates": agent_success_rates,
            "insights": [
                f"Network contains {total_agents} agents with {total_collaborations} collaboration pairs",
                f"Network density: {density:.2f} (0=sparse, 1=complete)",
                f"Average collaboration success rate: {avg_success_rate:.2f}",
                f"Most collaborative agent: {most_collaborative[0]} ({most_collaborative[1]} events)"
            ]
        }

def generate_mock_data():
    """Generate realistic mock data for demonstration"""
    print("📊 Generating mock data for ML analytics demonstration...")
    
    # Generate system metrics
    system_metrics = []
    for i in range(60):  # Last hour of data
        timestamp = datetime.now() - timedelta(minutes=60-i)
        metrics = {
            'timestamp': timestamp.isoformat(),
            'coordination_quality': 0.65 + 0.2 * np.sin(i * 0.1) + np.random.normal(0, 0.05),
            'sync_performance': 0.75 + 0.15 * np.cos(i * 0.08) + np.random.normal(0, 0.03),
            'system_health': 90 + 8 * np.sin(i * 0.05) + np.random.normal(0, 2),
            'performance_score': 75 + 20 * np.sin(i * 0.12) + np.random.normal(0, 3),
            'total_agents': 5,
            'active_agents': 5,
            'tasks_completed': i * 2 + np.random.randint(0, 5)
        }
        # Ensure values are within reasonable bounds
        metrics['coordination_quality'] = max(0, min(1, metrics['coordination_quality']))
        metrics['sync_performance'] = max(0, min(1, metrics['sync_performance']))
        metrics['system_health'] = max(0, min(100, metrics['system_health']))
        metrics['performance_score'] = max(0, min(100, metrics['performance_score']))
        
        system_metrics.append(metrics)
    
    # Generate agent data
    agent_names = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
    specializations = ['Analysis', 'Coordination', 'Problem Solving', 'Pattern Recognition', 'Decision Making']
    
    agents_data = []
    for i, (agent_id, spec) in enumerate(zip(agent_names, specializations)):
        # Create some variation in agent performance
        base_performance = 0.5 + (i * 0.1)  # Different performance levels
        for j in range(10):  # Multiple data points per agent
            agent = {
                'agent_id': agent_id,
                'performance_score': max(0, min(1, base_performance + np.random.normal(0, 0.1))),
                'tasks_completed': np.random.randint(8, 35),
                'collaboration_count': np.random.randint(3, 18),
                'active': True,
                'specialization': spec,
                'timestamp': (datetime.now() - timedelta(minutes=np.random.randint(1, 60))).isoformat()
            }
            agents_data.append(agent)
    
    # Generate collaboration events
    events_data = []
    event_types = ['Task Distribution', 'Data Sharing', 'Decision Making', 'Problem Solving', 'Analysis']
    outcomes = ['Success', 'Partial Success', 'Optimization Achieved', 'Knowledge Shared']
    
    for i in range(30):
        participants = np.random.choice(agent_names, size=np.random.randint(2, 4), replace=False).tolist()
        event = {
            'timestamp': (datetime.now() - timedelta(minutes=np.random.randint(1, 120))).isoformat(),
            'event_type': np.random.choice(event_types),
            'participants': participants,
            'success': np.random.random() > 0.15,  # 85% success rate
            'duration_seconds': np.random.uniform(0.5, 25.0),
            'outcome': np.random.choice(outcomes)
        }
        events_data.append(event)
    
    return system_metrics, agents_data, events_data

def run_ml_analytics_demo():
    """Run comprehensive ML analytics demonstration"""
    print("\n" + "="*80)
    print("🧠 KAIROS ML ANALYTICS DEMONSTRATION 🧠")
    print("="*80)
    
    # Initialize analytics engine
    ml_engine = SimplifiedMLAnalytics()
    
    # Generate and load mock data
    system_metrics, agents_data, events_data = generate_mock_data()
    
    ml_engine.add_system_metrics(system_metrics)
    ml_engine.add_agent_data(agents_data)
    ml_engine.add_collaboration_events(events_data)
    
    print(f"✅ Loaded {len(system_metrics)} system metrics, {len(agents_data)} agent records, {len(events_data)} events")
    
    # Run analytics
    results = {}
    
    print("\n📊 Running ML Analytics Suite...")
    
    # Performance Clustering
    results['clustering'] = ml_engine.analyze_performance_clusters()
    
    # Anomaly Detection
    results['anomaly_detection'] = ml_engine.detect_anomalies()
    
    # Performance Prediction
    results['predictions'] = ml_engine.predict_performance()
    
    # Collaboration Network Analysis
    results['collaboration_network'] = ml_engine.analyze_collaboration_network()
    
    # Display Results
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    # Clustering Results
    print("\n🎯 PERFORMANCE CLUSTERING:")
    clustering = results['clustering']
    if 'clusters' in clustering:
        for cluster_name, cluster_info in clustering['clusters'].items():
            print(f"   {cluster_name}: {cluster_info['performance_level']}")
            print(f"     Agents: {', '.join(cluster_info['agents'])}")
            print(f"     Avg Performance: {cluster_info['avg_performance']:.2f}")
        for insight in clustering['insights']:
            print(f"   • {insight}")
    else:
        print(f"   {clustering['message']}")
    
    # Anomaly Detection Results
    print("\n🚨 ANOMALY DETECTION:")
    anomaly = results['anomaly_detection']
    if 'insights' in anomaly:
        for insight in anomaly['insights']:
            print(f"   • {insight}")
    else:
        print(f"   {anomaly['message']}")
    
    # Prediction Results
    print("\n🔮 PERFORMANCE PREDICTIONS:")
    predictions = results['predictions']
    if 'predictions' in predictions:
        for metric, pred_data in predictions['predictions'].items():
            print(f"   {metric.replace('_', ' ').title()}: {pred_data['trend']} trend")
            print(f"     Accuracy: {pred_data['accuracy_score']:.2f}")
        for insight in predictions['insights']:
            print(f"   • {insight}")
    else:
        print(f"   {predictions['message']}")
    
    # Collaboration Network Results
    print("\n🕸️ COLLABORATION NETWORK:")
    network = results['collaboration_network']
    if 'insights' in network:
        for insight in network['insights']:
            print(f"   • {insight}")
    else:
        print(f"   {network['message']}")
    
    # Generate Automated Insights
    print("\n" + "="*60)
    print("🤖 AUTOMATED INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    insights = []
    recommendations = []
    
    # Generate insights based on results
    if 'clusters' in results['clustering']:
        num_clusters = len(results['clustering']['clusters'])
        insights.append(f"🎯 Identified {num_clusters} distinct agent performance groups")
    
    if 'anomaly_percentage' in results['anomaly_detection']:
        anomaly_rate = results['anomaly_detection']['anomaly_percentage']
        if anomaly_rate < 3:
            insights.append("✅ System stability excellent - minimal anomalies detected")
        elif anomaly_rate < 8:
            insights.append("⚠️ Moderate anomaly activity - monitor system performance")
        else:
            insights.append("🚨 High anomaly rate - investigate system issues")
    
    if 'predictions' in results['predictions']:
        num_predictions = len(results['predictions']['predictions'])
        insights.append(f"🔮 Generated performance forecasts for {num_predictions} key metrics")
    
    if 'network_density' in results['collaboration_network']:
        density = results['collaboration_network']['network_density']
        success_rate = results['collaboration_network']['average_success_rate']
        insights.append(f"🕸️ Collaboration network density: {density:.2f}, Success rate: {success_rate:.2f}")
    
    # Generate recommendations
    if 'clusters' in results['clustering']:
        recommendations.append("🎯 Optimize task allocation based on agent performance clusters")
        recommendations.append("📈 Focus training on lower-performing agent clusters")
    
    if 'network_density' in results['collaboration_network']:
        density = results['collaboration_network']['network_density']
        if density < 0.3:
            recommendations.append("🤝 Increase cross-agent collaboration opportunities")
        elif density > 0.8:
            recommendations.append("⚡ Consider agent specialization to improve efficiency")
    
    recommendations.extend([
        "📊 Continue regular ML analytics monitoring",
        "🔄 Use predictions for proactive system optimization",
        "🎯 Leverage collaboration patterns for team formation"
    ])
    
    # Display insights and recommendations
    print("\n💡 Key Insights:")
    for insight in insights:
        print(f"   {insight}")
    
    print("\n🚀 Optimization Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "="*80)
    print("✅ ML Analytics demonstration completed successfully!")
    print("="*80)
    
    return ml_engine, results

if __name__ == "__main__":
    # Run the demonstration
    ml_engine, results = run_ml_analytics_demo()