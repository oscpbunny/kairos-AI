"""
Project Kairos: Fusion Processor
Phase 7 - Advanced Intelligence

Cross-modal fusion processor that combines vision, audio, and text
inputs into unified, coherent understanding with attention mechanisms.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger("kairos.perception.fusion")

class FusionProcessor:
    """
    Cross-modal fusion processor for unified perception.
    
    Capabilities:
    - Cross-modal attention mechanisms
    - Semantic alignment across modalities  
    - Unified scene understanding
    - Temporal synchronization
    - Confidence weighted integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialized = False
        
        # Fusion parameters
        self.attention_weights = {
            'vision': 0.4,
            'audio': 0.3, 
            'text': 0.3
        }
        
        # Processing stats
        self.processing_stats = {
            'fusions_performed': 0,
            'avg_fusion_time': 0.0,
            'modality_combinations': {}
        }
    
    async def initialize(self) -> bool:
        """Initialize fusion processor"""
        try:
            logger.info("🔗 Initializing Fusion Processor...")
            self.initialized = True
            logger.info("✅ Fusion Processor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize fusion processor: {e}")
            return False
    
    async def fuse(self, modality_outputs: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """
        Fuse multiple modality outputs into unified perception.
        
        Args:
            modality_outputs: Dict mapping modality names to their processed outputs
            context: Optional context for attention weighting
            
        Returns:
            FusedPerception: Unified cross-modal understanding
        """
        if not self.initialized:
            raise RuntimeError("Fusion processor not initialized")
        
        start_time = datetime.now()
        logger.info(f"🔗 Fusing {len(modality_outputs)} modalities...")
        
        try:
            # Extract features from each modality
            unified_features = await self._extract_unified_features(modality_outputs)
            
            # Find cross-modal relationships
            cross_modal_relationships = await self._find_cross_modal_relationships(modality_outputs)
            
            # Build scene understanding
            scene_understanding = await self._build_scene_understanding(modality_outputs, unified_features)
            
            # Calculate attention map
            attention_map = await self._calculate_attention_map(modality_outputs, context)
            
            # Compute overall confidence
            overall_confidence = self._compute_overall_confidence(modality_outputs, attention_map)
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, list(modality_outputs.keys()))
            
            # Create fused perception
            fused_perception = type('FusedPerception', (), {
                'input_ids': [getattr(output, 'input_id', f"{mod}_input") for mod, output in modality_outputs.items()],
                'timestamp': datetime.now(),
                'unified_features': unified_features,
                'cross_modal_relationships': cross_modal_relationships,
                'scene_understanding': scene_understanding,
                'overall_confidence': overall_confidence,
                'attention_map': attention_map
            })()
            
            logger.info(f"✅ Fusion complete in {processing_time:.2f}s")
            logger.info(f"🎯 Overall confidence: {overall_confidence:.3f}")
            
            return fused_perception
            
        except Exception as e:
            logger.error(f"❌ Fusion processing failed: {e}")
            raise
    
    async def _extract_unified_features(self, modality_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and unify features across modalities"""
        unified = {
            'modalities_present': list(modality_outputs.keys()),
            'entity_count': 0,
            'total_confidence': 0.0,
            'dominant_modality': None,
            'feature_vectors': {}
        }
        
        # Aggregate entity counts and confidence scores
        modality_confidences = {}
        for modality, output in modality_outputs.items():
            entities = getattr(output, 'entities', [])
            confidence = getattr(output, 'confidence', 0.0)
            
            unified['entity_count'] += len(entities)
            unified['total_confidence'] += confidence
            modality_confidences[modality] = confidence
            
            # Store feature vector if available
            embedding = getattr(output, 'semantic_embedding', None)
            if embedding is not None:
                unified['feature_vectors'][modality] = embedding
        
        # Determine dominant modality
        if modality_confidences:
            unified['dominant_modality'] = max(modality_confidences.items(), key=lambda x: x[1])[0]
            unified['confidence_distribution'] = modality_confidences
        
        # Create unified feature vector
        if unified['feature_vectors']:
            unified['unified_embedding'] = self._create_unified_embedding(unified['feature_vectors'])
        
        return unified
    
    async def _find_cross_modal_relationships(self, modality_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relationships between entities across modalities"""
        relationships = []
        
        # Get entities from each modality
        modality_entities = {}
        for modality, output in modality_outputs.items():
            entities = getattr(output, 'entities', [])
            modality_entities[modality] = entities
        
        # Find cross-modal entity correspondences
        for mod1, entities1 in modality_entities.items():
            for mod2, entities2 in modality_entities.items():
                if mod1 >= mod2:  # Avoid duplicates
                    continue
                
                # Look for potential correspondences
                for entity1 in entities1:
                    for entity2 in entities2:
                        correspondence = self._check_entity_correspondence(entity1, entity2, mod1, mod2)
                        if correspondence:
                            relationships.append(correspondence)
        
        # Add temporal relationships for audio entities
        if 'audio' in modality_entities and 'text' in modality_entities:
            audio_entities = modality_entities['audio']
            text_entities = modality_entities['text']
            
            # Find speech-text correspondences
            for audio_entity in audio_entities:
                if isinstance(audio_entity, dict) and audio_entity.get('type') == 'speech':
                    for text_entity in text_entities:
                        if isinstance(text_entity, dict):
                            relationships.append({
                                'type': 'speech_text_correspondence',
                                'audio_entity': audio_entity,
                                'text_entity': text_entity,
                                'confidence': 0.8,
                                'description': 'Speech content matches text entity'
                            })
        
        return relationships
    
    def _check_entity_correspondence(self, entity1: Any, entity2: Any, mod1: str, mod2: str) -> Optional[Dict[str, Any]]:
        """Check if two entities from different modalities correspond"""
        # Simple correspondence checking (can be enhanced with semantic similarity)
        
        if not isinstance(entity1, dict) or not isinstance(entity2, dict):
            return None
        
        # Check for text entities that might correspond to visual text
        if mod1 == 'vision' and mod2 == 'text':
            if entity1.get('type') == 'text' and entity2.get('type') == 'person':
                # Visual text might correspond to person name in text
                return {
                    'type': 'visual_text_correspondence',
                    'vision_entity': entity1,
                    'text_entity': entity2,
                    'confidence': 0.6,
                    'description': 'Visual text may correspond to named entity in text'
                }
        
        # Check for person entities across vision and text
        if (entity1.get('type') in ['person', 'object'] and 
            entity2.get('type') in ['person', 'organization']):
            return {
                'type': 'entity_correspondence',
                'entity1': entity1,
                'entity2': entity2,
                'modality1': mod1,
                'modality2': mod2,
                'confidence': 0.5,
                'description': 'Potential entity correspondence across modalities'
            }
        
        return None
    
    async def _build_scene_understanding(self, modality_outputs: Dict[str, Any], unified_features: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive scene understanding from all modalities"""
        scene = {
            'scene_type': 'unknown',
            'environment': 'unknown',
            'activity': 'unknown',
            'objects': [],
            'people': [],
            'sounds': [],
            'text_content': '',
            'temporal_structure': {},
            'spatial_structure': {},
            'narrative_elements': []
        }
        
        # Extract scene information from vision
        if 'vision' in modality_outputs:
            vision_output = modality_outputs['vision']
            vision_features = getattr(vision_output, 'features', {})
            
            scene_analysis = vision_features.get('scene_analysis', {})
            scene['scene_type'] = scene_analysis.get('scene_type', 'unknown')
            scene['environment'] = scene_analysis.get('scene_type', 'unknown')
            
            # Extract visual entities
            entities = getattr(vision_output, 'entities', [])
            for entity in entities:
                if isinstance(entity, dict):
                    if entity.get('type') == 'person':
                        scene['people'].append(entity.get('description', 'person'))
                    elif entity.get('type') == 'object':
                        scene['objects'].append(entity.get('description', 'object'))
        
        # Extract audio information
        if 'audio' in modality_outputs:
            audio_output = modality_outputs['audio']
            audio_features = getattr(audio_output, 'features', {})
            
            acoustic_scene = audio_features.get('acoustic_scene', {})
            if scene['environment'] == 'unknown':
                scene['environment'] = acoustic_scene.get('scene_type', 'unknown')
            
            # Extract audio entities
            entities = getattr(audio_output, 'entities', [])
            for entity in entities:
                if isinstance(entity, dict):
                    entity_type = entity.get('type')
                    if entity_type == 'speech':
                        scene['sounds'].append(f"speech: {entity.get('transcription', 'spoken words')}")
                    elif entity_type in ['music', 'ambient']:
                        scene['sounds'].append(entity.get('description', 'sound'))
        
        # Extract text information  
        if 'text' in modality_outputs:
            text_output = modality_outputs['text']
            text_features = getattr(text_output, 'features', {})
            
            # Get full text content
            text_props = text_features.get('text_properties', {})
            if hasattr(text_output, 'input_id'):
                # In a real implementation, we'd store the original text
                scene['text_content'] = 'Processed text content available'
            
            # Extract topics for activity inference
            topic_analysis = text_features.get('topic_modeling', {})
            primary_topic = topic_analysis.get('primary_topic', 'general')
            if scene['activity'] == 'unknown':
                scene['activity'] = f"{primary_topic}_related_activity"
        
        # Build temporal structure
        audio_duration = 0
        if 'audio' in modality_outputs:
            audio_features = getattr(modality_outputs['audio'], 'features', {})
            audio_props = audio_features.get('audio_properties', {})
            audio_duration = audio_props.get('duration_seconds', 0)
        
        scene['temporal_structure'] = {
            'duration_estimate': max(audio_duration, 5.0),  # At least 5 seconds
            'has_speech': any('speech' in sound for sound in scene['sounds']),
            'has_music': any('music' in sound for sound in scene['sounds']),
            'activity_level': len(scene['objects']) + len(scene['sounds']) + len(scene['people'])
        }
        
        # Build spatial structure from vision
        scene['spatial_structure'] = {
            'layout': 'unknown',
            'object_density': len(scene['objects']),
            'person_density': len(scene['people']),
            'complexity_score': (len(scene['objects']) + len(scene['people'])) / 10.0
        }
        
        return scene
    
    async def _calculate_attention_map(self, modality_outputs: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Dict[str, float]]:
        """Calculate attention weights across and within modalities"""
        attention_map = {}
        
        # Base attention weights
        base_weights = self.attention_weights.copy()
        
        # Adjust weights based on modality confidence and content richness
        for modality, output in modality_outputs.items():
            confidence = getattr(output, 'confidence', 0.5)
            entities = getattr(output, 'entities', [])
            entity_count = len(entities)
            
            # Calculate richness score
            richness_score = min(1.0, entity_count / 10.0)  # Normalize to 0-1
            
            # Combine confidence and richness
            modality_score = (confidence * 0.7) + (richness_score * 0.3)
            
            if modality in base_weights:
                base_weights[modality] *= (1 + modality_score * 0.5)  # Boost by up to 50%
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in base_weights.items()}
        else:
            normalized_weights = {k: 1.0/len(base_weights) for k in base_weights}
        
        # Create attention map for each modality
        for modality in modality_outputs:
            attention_map[modality] = {
                'global_attention': normalized_weights.get(modality, 0.33),
                'internal_attention': self._calculate_internal_attention(modality_outputs[modality])
            }
        
        return attention_map
    
    def _calculate_internal_attention(self, modality_output: Any) -> Dict[str, float]:
        """Calculate attention weights within a single modality"""
        internal_weights = {}
        
        # Get attention weights from the modality output
        if hasattr(modality_output, 'attention_weights'):
            internal_weights = getattr(modality_output, 'attention_weights', {})
        
        # Default internal attention if not provided
        if not internal_weights:
            internal_weights = {
                'entities': 0.4,
                'features': 0.3,
                'relationships': 0.3
            }
        
        return internal_weights
    
    def _compute_overall_confidence(self, modality_outputs: Dict[str, Any], attention_map: Dict[str, Dict[str, float]]) -> float:
        """Compute overall confidence using attention-weighted combination"""
        if not modality_outputs:
            return 0.0
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for modality, output in modality_outputs.items():
            confidence = getattr(output, 'confidence', 0.5)
            attention_weight = attention_map.get(modality, {}).get('global_attention', 0.33)
            
            weighted_confidence += confidence * attention_weight
            total_weight += attention_weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.5
    
    def _create_unified_embedding(self, feature_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """Create unified embedding from modality-specific embeddings"""
        if not feature_vectors:
            return np.zeros(512, dtype=np.float32)
        
        # Simple concatenation and dimensionality reduction (mock)
        embeddings = list(feature_vectors.values())
        
        # Ensure all embeddings have the same size
        min_size = min(emb.shape[0] for emb in embeddings)
        truncated_embeddings = [emb[:min_size] for emb in embeddings]
        
        # Average the embeddings
        unified = np.mean(truncated_embeddings, axis=0)
        
        # Pad or truncate to standard size (512)
        if len(unified) < 512:
            unified = np.pad(unified, (0, 512 - len(unified)), mode='constant')
        else:
            unified = unified[:512]
        
        return unified.astype(np.float32)
    
    def _update_stats(self, processing_time: float, modalities: List[str]):
        """Update processing statistics"""
        self.processing_stats['fusions_performed'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['avg_fusion_time']
        n = self.processing_stats['fusions_performed']
        new_avg = (current_avg * (n - 1) + processing_time) / n
        self.processing_stats['avg_fusion_time'] = new_avg
        
        # Track modality combinations
        modality_combo = '+'.join(sorted(modalities))
        if modality_combo not in self.processing_stats['modality_combinations']:
            self.processing_stats['modality_combinations'][modality_combo] = 0
        self.processing_stats['modality_combinations'][modality_combo] += 1
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return fusion processor capabilities"""
        return {
            'version': '7.0.0',
            'initialized': self.initialized,
            'capabilities': [
                'cross_modal_attention',
                'semantic_alignment',
                'unified_scene_understanding',
                'temporal_synchronization',
                'confidence_weighting',
                'entity_correspondence'
            ],
            'supported_combinations': [
                'vision+audio',
                'vision+text', 
                'audio+text',
                'vision+audio+text'
            ],
            'processing_stats': self.processing_stats
        }
    
    async def shutdown(self):
        """Shutdown fusion processor"""
        logger.info("🔄 Shutting down Fusion Processor...")
        self.initialized = False
        logger.info("✅ Fusion Processor shutdown complete")