"""
Project Kairos: Text Processor
Phase 7 - Advanced Intelligence

Text processing module enabling advanced natural language understanding,
document analysis, code comprehension, and knowledge extraction.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import re
import json

logger = logging.getLogger("kairos.perception.text")

@dataclass
class TextEntity:
    """Represents a detected text entity"""
    entity_type: str  # 'person', 'organization', 'location', 'concept', 'code', 'keyword'
    confidence: float
    position: Tuple[int, int]  # start_char, end_char
    text: str
    attributes: Dict[str, Any]
    description: str

class TextProcessor:
    """
    Advanced text processor for multi-modal perception.
    
    Capabilities:
    - Named entity recognition
    - Sentiment analysis  
    - Topic modeling
    - Code analysis
    - Knowledge extraction
    - Document structure analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialized = False
        
        # Processing stats
        self.processing_stats = {
            'texts_processed': 0,
            'entities_extracted': 0,
            'avg_processing_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize text processing components"""
        try:
            logger.info("📝 Initializing Text Processor...")
            self.initialized = True
            logger.info("✅ Text Processor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize text processor: {e}")
            return False
    
    async def process(self, input_data) -> Any:
        """Process text input and return structured understanding"""
        if not self.initialized:
            raise RuntimeError("Text processor not initialized")
        
        start_time = datetime.now()
        logger.info(f"📝 Processing text input: {input_data.input_id}")
        
        try:
            text = str(input_data.data)
            
            # Analyze text concurrently
            tasks = [
                self._extract_entities(text),
                self._analyze_sentiment(text),
                self._extract_topics(text),
                self._analyze_structure(text)
            ]
            
            entity_results, sentiment_results, topic_results, structure_results = await asyncio.gather(*tasks)
            
            # Combine results
            entities = entity_results.get('entities', [])
            
            features = {
                'text_properties': self._analyze_text_properties(text),
                'entity_extraction': entity_results,
                'sentiment_analysis': sentiment_results,
                'topic_modeling': topic_results,
                'structure_analysis': structure_results
            }
            
            relationships = self._analyze_entity_relationships(entities)
            overall_confidence = 0.85  # Mock
            attention_weights = self._calculate_attention_weights(entities, features)
            semantic_embedding = self._create_semantic_embedding(text, features)
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, len(entities))
            
            # Create output
            output = type('PerceptionOutput', (), {
                'input_id': input_data.input_id,
                'timestamp': datetime.now(),
                'modality': 'text',
                'features': features,
                'entities': [self._entity_to_dict(e) for e in entities],
                'relationships': relationships,
                'confidence': overall_confidence,
                'attention_weights': attention_weights,
                'semantic_embedding': semantic_embedding
            })()
            
            logger.info(f"✅ Text processing complete in {processing_time:.2f}s")
            return output
            
        except Exception as e:
            logger.error(f"❌ Text processing failed: {e}")
            raise
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text"""
        entities = []
        
        # Simple regex-based entity extraction (mock)
        patterns = {
            'person': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?\S+)?)?',
            'number': r'\b\d+\.?\d*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity = TextEntity(
                    entity_type=entity_type,
                    confidence=0.8,
                    position=(match.start(), match.end()),
                    text=match.group(),
                    attributes={'pattern': pattern},
                    description=f"Detected {entity_type}"
                )
                entities.append(entity)
        
        return {
            'entities': entities,
            'total_entities': len(entities),
            'entity_types': list(set(e.entity_type for e in entities))
        }
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        # Simple sentiment analysis (mock)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = 0.7
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = -0.7
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'confidence': 0.75
        }
    
    async def _extract_topics(self, text: str) -> Dict[str, Any]:
        """Extract topics and themes from text"""
        # Simple topic extraction (mock)
        topic_keywords = {
            'technology': ['ai', 'computer', 'software', 'digital', 'data', 'algorithm'],
            'business': ['market', 'company', 'revenue', 'profit', 'customer', 'sales'],
            'science': ['research', 'study', 'analysis', 'experiment', 'discovery'],
            'health': ['medical', 'patient', 'treatment', 'disease', 'healthcare']
        }
        
        words = text.lower().split()
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for word in words if word in keywords)
            if score > 0:
                topic_scores[topic] = score / len(words)
        
        primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0] if topic_scores else 'general'
        
        return {
            'primary_topic': primary_topic,
            'topic_scores': topic_scores,
            'topics_detected': list(topic_scores.keys()),
            'topic_confidence': max(topic_scores.values()) if topic_scores else 0.5
        }
    
    async def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and organization"""
        lines = text.split('\n')
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        # Count structural elements
        headers = len([line for line in lines if line.strip() and line.strip()[0] == '#'])
        bullet_points = len([line for line in lines if line.strip().startswith(('*', '-', '•'))])
        
        return {
            'line_count': len(lines),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'header_count': headers,
            'bullet_point_count': bullet_points,
            'structure_type': 'structured' if headers > 0 or bullet_points > 0 else 'prose',
            'readability_score': min(100, max(0, 100 - len(text.split()) / 10))  # Simple mock
        }
    
    def _analyze_text_properties(self, text: str) -> Dict[str, Any]:
        """Analyze basic text properties"""
        words = text.split()
        chars = len(text)
        
        return {
            'character_count': chars,
            'word_count': len(words),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'language': 'en',  # Mock
            'encoding': 'utf-8'
        }
    
    def _analyze_entity_relationships(self, entities: List[TextEntity]) -> List[Dict[str, Any]]:
        """Analyze relationships between entities"""
        relationships = []
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Simple proximity-based relationship
                distance = abs(entity1.position[0] - entity2.position[0])
                if distance < 100:  # Within 100 characters
                    relationships.append({
                        'entity1': entity1.text,
                        'entity2': entity2.text,
                        'relationship': 'near',
                        'distance': distance,
                        'confidence': 0.6
                    })
        
        return relationships
    
    def _calculate_attention_weights(self, entities: List[TextEntity], features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate attention weights for text elements"""
        weights = {
            'entities': 0.3,
            'sentiment': 0.2,
            'topics': 0.3,
            'structure': 0.2
        }
        
        # Adjust based on content
        if len(entities) > 10:
            weights['entities'] = 0.5
        
        if abs(features.get('sentiment_analysis', {}).get('score', 0)) > 0.5:
            weights['sentiment'] = 0.4
        
        return weights
    
    def _create_semantic_embedding(self, text: str, features: Dict[str, Any]) -> np.ndarray:
        """Create semantic embedding for text"""
        # Mock embedding based on text features
        embedding_size = 512
        embedding = np.random.randn(embedding_size) * 0.1
        
        # Adjust based on content
        word_count = len(text.split())
        embedding[0] = word_count / 1000.0
        
        sentiment_score = features.get('sentiment_analysis', {}).get('score', 0)
        embedding[1] = sentiment_score
        
        return embedding.astype(np.float32)
    
    def _entity_to_dict(self, entity: TextEntity) -> Dict[str, Any]:
        """Convert TextEntity to dictionary"""
        return {
            'type': entity.entity_type,
            'confidence': entity.confidence,
            'position': entity.position,
            'text': entity.text,
            'attributes': entity.attributes,
            'description': entity.description
        }
    
    def _update_stats(self, processing_time: float, entity_count: int):
        """Update processing statistics"""
        self.processing_stats['texts_processed'] += 1
        self.processing_stats['entities_extracted'] += entity_count
        
        current_avg = self.processing_stats['avg_processing_time']
        n = self.processing_stats['texts_processed']
        new_avg = (current_avg * (n - 1) + processing_time) / n
        self.processing_stats['avg_processing_time'] = new_avg
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return text processing capabilities"""
        return {
            'version': '7.0.0',
            'initialized': self.initialized,
            'capabilities': [
                'entity_extraction',
                'sentiment_analysis',
                'topic_modeling',
                'structure_analysis',
                'keyword_extraction',
                'language_detection'
            ],
            'processing_stats': self.processing_stats
        }
    
    async def shutdown(self):
        """Shutdown text processor"""
        logger.info("🔄 Shutting down Text Processor...")
        self.initialized = False
        logger.info("✅ Text Processor shutdown complete")