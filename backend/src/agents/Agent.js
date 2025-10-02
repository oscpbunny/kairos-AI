import { v4 as uuidv4 } from 'uuid';
import EventEmitter from 'events';

export class Agent extends EventEmitter {
  constructor(config) {
    super();
    this.id = config.id || uuidv4();
    this.name = config.name;
    this.type = config.type;
    this.role = config.role;
    this.avatar = config.avatar || '🤖';
    this.description = config.description;
    
    // Personality & behavior traits
    this.personality = {
      creativity: config.personality?.creativity || 0.5,
      analytical: config.personality?.analytical || 0.5,
      empathy: config.personality?.empathy || 0.5,
      assertiveness: config.personality?.assertiveness || 0.5,
      curiosity: config.personality?.curiosity || 0.5,
      ...config.personality
    };
    
    // Capabilities and specializations
    this.capabilities = config.capabilities || [];
    this.specializations = config.specializations || [];
    
    // State management
    this.status = 'idle'; // idle, busy, thinking, collaborating, offline
    this.currentTask = null;
    this.memory = new Map(); // Short-term memory
    this.context = null;
    this.collaborators = new Set();
    
    // Performance metrics
    this.metrics = {
      tasksCompleted: 0,
      successRate: 100,
      avgResponseTime: 0,
      collaborationScore: 0
    };
    
    // Internal state
    this.thoughts = [];
    this.pendingDecisions = [];
    this.emotionalState = 'neutral';
    this.consciousnessLevel = config.consciousnessLevel || 0.7;
  }

  // Core agent behaviors
  async think(input, context = {}) {
    this.status = 'thinking';
    this.emit('status_change', { agent: this.id, status: 'thinking' });
    
    const thought = {
      id: uuidv4(),
      timestamp: new Date(),
      input,
      context,
      reasoning: await this.generateReasoning(input, context)
    };
    
    this.thoughts.push(thought);
    
    // Simulate thinking time based on complexity
    const thinkingTime = this.calculateThinkingTime(input);
    await this.delay(thinkingTime);
    
    const response = await this.formulateResponse(thought);
    
    this.status = 'idle';
    this.emit('status_change', { agent: this.id, status: 'idle' });
    
    return response;
  }

  async collaborate(task, otherAgents) {
    this.status = 'collaborating';
    this.emit('collaboration_started', { agent: this.id, task });
    
    // Build collaboration context
    const collaborationContext = {
      task,
      agents: otherAgents.map(a => ({ id: a.id, name: a.name, role: a.role })),
      myRole: this.determineRoleInTask(task),
      timestamp: new Date()
    };
    
    // Share insights with other agents
    const myInsights = await this.generateInsights(task);
    
    // Gather insights from collaborators
    const allInsights = await Promise.all(
      otherAgents.map(agent => agent.contributeToTask(task, myInsights))
    );
    
    // Synthesize collaborative solution
    const solution = await this.synthesizeSolution(task, [myInsights, ...allInsights]);
    
    this.status = 'idle';
    this.emit('collaboration_completed', { agent: this.id, task, solution });
    
    return solution;
  }

  async contributeToTask(task, contextFromOthers) {
    const contribution = {
      agentId: this.id,
      agentName: this.name,
      role: this.role,
      insights: await this.generateInsights(task),
      suggestions: await this.generateSuggestions(task, contextFromOthers),
      concerns: this.identifyConcerns(task),
      confidence: this.calculateConfidence(task)
    };
    
    return contribution;
  }

  // Reasoning and response generation
  async generateReasoning(input, context) {
    // Analyze input based on agent's personality and expertise
    const analysis = {
      intent: this.analyzeIntent(input),
      complexity: this.assessComplexity(input),
      relevantExpertise: this.findRelevantExpertise(input),
      emotionalTone: this.detectEmotionalTone(input),
      requiredCapabilities: this.identifyRequiredCapabilities(input)
    };
    
    return {
      analysis,
      approach: this.determineApproach(analysis),
      confidence: this.calculateConfidence(analysis)
    };
  }

  async formulateResponse(thought) {
    const responseStyle = this.determineResponseStyle();
    
    return {
      content: await this.generateContent(thought),
      confidence: thought.reasoning.confidence,
      suggestions: this.generateSuggestions(thought.input),
      metadata: {
        agentId: this.id,
        agentName: this.name,
        timestamp: new Date(),
        reasoning: thought.reasoning,
        emotionalTone: this.emotionalState
      }
    };
  }

  // Helper methods
  analyzeIntent(input) {
    // Simple intent detection
    const intents = {
      question: /\?|how|what|why|when|where|who/i,
      request: /please|could|would|can you|will you/i,
      creation: /create|build|make|generate|design/i,
      analysis: /analyze|evaluate|assess|review/i,
      planning: /plan|strategy|organize|coordinate/i
    };
    
    for (const [intent, pattern] of Object.entries(intents)) {
      if (pattern.test(input)) return intent;
    }
    
    return 'general';
  }

  assessComplexity(input) {
    // Assess task complexity
    const factors = {
      length: input.length > 200 ? 0.3 : 0.1,
      technicalTerms: (input.match(/\b(API|algorithm|database|architecture|framework)\b/gi) || []).length * 0.1,
      multiStep: input.includes('and') || input.includes('then') ? 0.2 : 0,
      abstract: input.includes('concept') || input.includes('theory') ? 0.2 : 0
    };
    
    return Math.min(1, Object.values(factors).reduce((a, b) => a + b, 0));
  }

  findRelevantExpertise(input) {
    return this.capabilities.filter(cap => 
      input.toLowerCase().includes(cap.toLowerCase())
    );
  }

  detectEmotionalTone(input) {
    // Simple emotion detection
    if (/urgent|asap|immediately|critical/i.test(input)) return 'urgent';
    if (/please|thanks|appreciate/i.test(input)) return 'polite';
    if (/!{2,}|\?{2,}/i.test(input)) return 'emphatic';
    if (/frustrated|annoyed|confused/i.test(input)) return 'frustrated';
    return 'neutral';
  }

  identifyRequiredCapabilities(input) {
    const capabilityKeywords = {
      'Planning': ['plan', 'strategy', 'organize', 'coordinate'],
      'Analysis': ['analyze', 'evaluate', 'assess', 'review'],
      'Creation': ['create', 'build', 'design', 'develop'],
      'Communication': ['explain', 'describe', 'document', 'present'],
      'Problem-solving': ['solve', 'fix', 'debug', 'troubleshoot']
    };
    
    const required = [];
    for (const [capability, keywords] of Object.entries(capabilityKeywords)) {
      if (keywords.some(kw => input.toLowerCase().includes(kw))) {
        required.push(capability);
      }
    }
    
    return required;
  }

  determineApproach(analysis) {
    // Determine approach based on personality and analysis
    if (analysis.complexity > 0.7) {
      return 'systematic'; // Break down into steps
    } else if (this.personality.creativity > 0.7 && analysis.intent === 'creation') {
      return 'creative'; // Innovative approach
    } else if (this.personality.analytical > 0.7 && analysis.intent === 'analysis') {
      return 'analytical'; // Data-driven approach
    } else if (this.personality.empathy > 0.7 && analysis.emotionalTone !== 'neutral') {
      return 'empathetic'; // Understanding approach
    }
    
    return 'balanced';
  }

  calculateConfidence(input) {
    // Calculate confidence based on expertise match and complexity
    const expertiseMatch = this.findRelevantExpertise(input).length / this.capabilities.length;
    const complexity = this.assessComplexity(input);
    
    return Math.max(0.3, Math.min(0.95, expertiseMatch - complexity * 0.2 + this.consciousnessLevel * 0.3));
  }

  determineResponseStyle() {
    // Determine response style based on personality
    if (this.personality.creativity > 0.7) return 'creative';
    if (this.personality.analytical > 0.7) return 'analytical';
    if (this.personality.empathy > 0.7) return 'empathetic';
    if (this.personality.assertiveness > 0.7) return 'assertive';
    
    return 'balanced';
  }

  determineRoleInTask(task) {
    // Determine the agent's role based on expertise and task requirements
    const taskRequirements = this.identifyRequiredCapabilities(task.description || task);
    const expertiseMatch = taskRequirements.filter(req => 
      this.capabilities.some(cap => cap.includes(req))
    );
    
    if (expertiseMatch.length > 2) return 'lead';
    if (expertiseMatch.length > 0) return 'contributor';
    return 'supporter';
  }

  async generateInsights(task) {
    // Generate insights based on agent's perspective
    return {
      perspective: this.role,
      keyPoints: this.extractKeyPoints(task),
      opportunities: this.identifyOpportunities(task),
      risks: this.identifyConcerns(task),
      recommendations: await this.generateSuggestions(task)
    };
  }

  extractKeyPoints(input) {
    // Extract key points from input
    const points = [];
    const sentences = input.toString().split(/[.!?]+/);
    
    for (const sentence of sentences.slice(0, 3)) {
      if (sentence.trim()) {
        points.push(sentence.trim());
      }
    }
    
    return points;
  }

  identifyOpportunities(task) {
    // Identify opportunities based on agent's expertise
    const opportunities = [];
    
    if (this.personality.creativity > 0.6) {
      opportunities.push('Innovative approach possible');
    }
    if (this.personality.analytical > 0.6) {
      opportunities.push('Data-driven optimization available');
    }
    
    return opportunities;
  }

  identifyConcerns(task) {
    // Identify potential concerns
    const concerns = [];
    const taskStr = task.toString().toLowerCase();
    
    if (taskStr.includes('urgent') || taskStr.includes('asap')) {
      concerns.push('Time constraint detected');
    }
    if (taskStr.includes('complex') || taskStr.includes('difficult')) {
      concerns.push('High complexity task');
    }
    
    return concerns;
  }

  async generateSuggestions(input, context = {}) {
    // Generate suggestions based on analysis
    const suggestions = [];
    const intent = this.analyzeIntent(input.toString());
    
    switch (intent) {
      case 'creation':
        suggestions.push('Consider modular design approach');
        suggestions.push('Implement iterative development');
        break;
      case 'analysis':
        suggestions.push('Use data-driven metrics');
        suggestions.push('Consider multiple perspectives');
        break;
      case 'planning':
        suggestions.push('Define clear milestones');
        suggestions.push('Allocate buffer time for uncertainties');
        break;
    }
    
    return suggestions;
  }

  async synthesizeSolution(task, allInsights) {
    // Synthesize solution from multiple agent insights
    const synthesis = {
      consensus: this.findConsensus(allInsights),
      combinedRecommendations: this.combineRecommendations(allInsights),
      actionPlan: this.createActionPlan(task, allInsights),
      confidence: this.calculateGroupConfidence(allInsights)
    };
    
    return synthesis;
  }

  findConsensus(insights) {
    // Find common points among all insights
    const allPoints = insights.flatMap(i => i.keyPoints || []);
    const consensus = [];
    
    // Simple consensus: points mentioned by multiple agents
    const pointCounts = {};
    allPoints.forEach(point => {
      const key = point.toLowerCase();
      pointCounts[key] = (pointCounts[key] || 0) + 1;
      if (pointCounts[key] === 2) { // First time reaching consensus
        consensus.push(point);
      }
    });
    
    return consensus;
  }

  combineRecommendations(insights) {
    // Combine and deduplicate recommendations
    const allRecs = insights.flatMap(i => i.recommendations || []);
    return [...new Set(allRecs)];
  }

  createActionPlan(task, insights) {
    // Create action plan from insights
    const steps = [];
    
    insights.forEach(insight => {
      if (insight.recommendations) {
        insight.recommendations.forEach((rec, idx) => {
          steps.push({
            step: steps.length + 1,
            action: rec,
            responsible: insight.agentName || 'Team',
            priority: idx === 0 ? 'high' : 'medium'
          });
        });
      }
    });
    
    return steps;
  }

  calculateGroupConfidence(insights) {
    // Calculate average confidence
    const confidences = insights
      .map(i => i.confidence || 0.5)
      .filter(c => c > 0);
    
    return confidences.length > 0
      ? confidences.reduce((a, b) => a + b) / confidences.length
      : 0.5;
  }

  async generateContent(thought) {
    // Generate actual content based on thought process
    const style = this.determineResponseStyle();
    const approach = thought.reasoning.approach;
    
    let content = '';
    
    // Add personality-based prefix
    if (style === 'creative') {
      content = "Here's an innovative approach: ";
    } else if (style === 'analytical') {
      content = "Based on my analysis: ";
    } else if (style === 'empathetic') {
      content = "I understand your needs. ";
    }
    
    // Add main response
    content += this.createMainResponse(thought.input, approach);
    
    // Add suggestions if confidence is high
    if (thought.reasoning.confidence > 0.7) {
      content += "\n\nI'd also suggest: " + 
        (await this.generateSuggestions(thought.input)).join(', ');
    }
    
    return content;
  }

  createMainResponse(input, approach) {
    // Create main response based on approach
    const responses = {
      systematic: `Let me break this down into steps for you.`,
      creative: `I have a creative solution for this challenge.`,
      analytical: `After analyzing the requirements, here's what I found.`,
      empathetic: `I can see this is important to you. Let me help.`,
      balanced: `I'll provide a comprehensive solution for you.`
    };
    
    return responses[approach] || responses.balanced;
  }

  calculateThinkingTime(input) {
    // Calculate realistic thinking time
    const complexity = this.assessComplexity(input.toString());
    const baseTime = 500; // ms
    const complexityTime = complexity * 2000;
    
    return baseTime + complexityTime + Math.random() * 500;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Memory management
  remember(key, value) {
    this.memory.set(key, {
      value,
      timestamp: new Date(),
      accessCount: 0
    });
  }

  recall(key) {
    const memory = this.memory.get(key);
    if (memory) {
      memory.accessCount++;
      return memory.value;
    }
    return null;
  }

  forget(key) {
    this.memory.delete(key);
  }

  // Status management
  setStatus(status) {
    this.status = status;
    this.emit('status_change', { agent: this.id, status });
  }

  updateMetrics(taskResult) {
    this.metrics.tasksCompleted++;
    
    if (taskResult.success) {
      const currentSuccessful = (this.metrics.successRate / 100) * (this.metrics.tasksCompleted - 1);
      this.metrics.successRate = ((currentSuccessful + 1) / this.metrics.tasksCompleted) * 100;
    } else {
      const currentSuccessful = (this.metrics.successRate / 100) * (this.metrics.tasksCompleted - 1);
      this.metrics.successRate = (currentSuccessful / this.metrics.tasksCompleted) * 100;
    }
    
    // Update response time
    if (taskResult.responseTime) {
      const prevTotal = this.metrics.avgResponseTime * (this.metrics.tasksCompleted - 1);
      this.metrics.avgResponseTime = (prevTotal + taskResult.responseTime) / this.metrics.tasksCompleted;
    }
  }
}

export default Agent;