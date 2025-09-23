"""
Project Kairos: Multi-Modal Perception Engine
Phase 7 - Advanced Intelligence

Unified sensory processing framework enabling agents to perceive and understand
vision, audio, and text inputs with cross-modal fusion and attention mechanisms.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import base64
import io

# Core perception components
from .vision_processor import VisionProcessor
from .audio_processor import AudioProcessor  
from .text_processor import TextProcessor
from .fusion_processor import FusionProcessor

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

logger = logging.getLogger("kairos.perception.multimodal")

@dataclass
class PerceptionInput:
    """Unified input container for multi-modal data"""
    input_id: str
    timestamp: datetime
    source: str
    modality: str  # 'vision', 'audio', 'text'
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerceptionOutput:
    """Unified output container for processed multi-modal data"""
    input_id: str
    timestamp: datetime
    modality: str
    features: Dict[str, Any]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    confidence: float
    attention_weights: Dict[str, float]
    semantic_embedding: Optional[np.ndarray] = None

@dataclass
class FusedPerception:
    """Cross-modal fused understanding"""
    input_ids: List[str]
    timestamp: datetime
    unified_features: Dict[str, Any]
    cross_modal_relationships: List[Dict[str, Any]]
    scene_understanding: Dict[str, Any]
    overall_confidence: float
    attention_map: Dict[str, Dict[str, float]]

class PerceptionModality(ABC):
    """Abstract base class for perception modalities"""
    
    @abstractmethod
    async def process(self, input_data: PerceptionInput) -> PerceptionOutput:
        """Process input data for this modality"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities and supported formats"""
        pass

class MultiModalPerceptionEngine:
    """
    Advanced multi-modal perception engine with cross-modal fusion.
    
    Capabilities:
    - Vision: Image analysis, video processing, scene understanding
    - Audio: Speech recognition, sound analysis, music understanding
    - Text: NLP, document analysis, code understanding
    - Fusion: Cross-modal attention, unified representations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialized = False
        
        # Perception modality processors
        self.vision_processor = None
        self.audio_processor = None
        self.text_processor = None
        self.fusion_processor = None
        
        # Processing state
        self.active_inputs: Dict[str, PerceptionInput] = {}
        self.perception_history: List[FusedPerception] = []
        self.attention_context: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            'total_processed': 0,
            'processing_times': [],
            'accuracy_scores': [],
            'fusion_success_rate': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize all perception modalities"""
        try:
            logger.info("🧠 Initializing Multi-Modal Perception Engine...")
            
            # Initialize vision processor
            logger.info("👁️ Initializing Vision Processor...")
            self.vision_processor = VisionProcessor(self.config.get('vision', {}))
            await self.vision_processor.initialize()
            
            # Initialize audio processor
            logger.info("🎧 Initializing Audio Processor...")
            self.audio_processor = AudioProcessor(self.config.get('audio', {}))
            await self.audio_processor.initialize()
            
            # Initialize text processor
            logger.info("📝 Initializing Text Processor...")
            self.text_processor = TextProcessor(self.config.get('text', {}))
            await self.text_processor.initialize()
            
            # Initialize fusion processor
            logger.info("🔗 Initializing Fusion Processor...")
            self.fusion_processor = FusionProcessor(self.config.get('fusion', {}))
            await self.fusion_processor.initialize()
            
            self.initialized = True
            logger.info("✅ Multi-Modal Perception Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize perception engine: {e}")
            return False
    
    async def perceive(self, inputs: List[PerceptionInput]) -> FusedPerception:
        """
        Process multi-modal inputs and return fused perception.
        
        Args:
            inputs: List of perception inputs (vision, audio, text)
            
        Returns:
            FusedPerception: Unified understanding across all modalities
        """
        if not self.initialized:
            raise RuntimeError("Perception engine not initialized")
        
        start_time = datetime.now()
        logger.info(f"🔍 Processing {len(inputs)} multi-modal inputs...")
        
        try:
            # Process each modality independently
            modality_outputs = {}
            processing_tasks = []
            
            for input_data in inputs:
                if input_data.modality == 'vision' and self.vision_processor:
                    task = self.vision_processor.process(input_data)
                    processing_tasks.append(('vision', task))
                    
                elif input_data.modality == 'audio' and self.audio_processor:
                    task = self.audio_processor.process(input_data)
                    processing_tasks.append(('audio', task))
                    
                elif input_data.modality == 'text' and self.text_processor:
                    task = self.text_processor.process(input_data)
                    processing_tasks.append(('text', task))
            
            # Execute all processing tasks concurrently
            if processing_tasks:
                results = await asyncio.gather(*[task for _, task in processing_tasks])
                
                for (modality, _), result in zip(processing_tasks, results):
                    modality_outputs[modality] = result
            
            # Perform cross-modal fusion
            fused_perception = await self.fusion_processor.fuse(
                modality_outputs, 
                context=self.attention_context
            )
            
            # Update metrics and history
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics['total_processed'] += 1
            self.metrics['processing_times'].append(processing_time)
            self.perception_history.append(fused_perception)
            
            # Keep history bounded
            if len(self.perception_history) > 100:
                self.perception_history = self.perception_history[-50:]
            
            logger.info(f"✅ Multi-modal perception complete in {processing_time:.2f}s")
            logger.info(f"🎯 Overall confidence: {fused_perception.overall_confidence:.3f}")
            
            return fused_perception
            
        except Exception as e:
            logger.error(f"❌ Perception processing failed: {e}")
            raise
    
    async def perceive_vision(self, image_data: bytes, metadata: Dict[str, Any] = None) -> PerceptionOutput:
        """Process vision input (convenience method)"""
        input_data = PerceptionInput(
            input_id=f"vision_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            source="direct",
            modality="vision",
            data=image_data,
            metadata=metadata or {}
        )
        
        return await self.vision_processor.process(input_data)
    
    async def perceive_audio(self, audio_data: bytes, metadata: Dict[str, Any] = None) -> PerceptionOutput:
        """Process audio input (convenience method)"""
        input_data = PerceptionInput(
            input_id=f"audio_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            source="direct",
            modality="audio",
            data=audio_data,
            metadata=metadata or {}
        )
        
        return await self.audio_processor.process(input_data)
    
    async def perceive_text(self, text_data: str, metadata: Dict[str, Any] = None) -> PerceptionOutput:
        """Process text input (convenience method)"""
        input_data = PerceptionInput(
            input_id=f"text_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            source="direct",
            modality="text",
            data=text_data,
            metadata=metadata or {}
        )
        
        return await self.text_processor.process(input_data)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return comprehensive capabilities across all modalities"""
        capabilities = {
            'version': '7.0.0',
            'initialized': self.initialized,
            'modalities': {}
        }
        
        if self.vision_processor:
            capabilities['modalities']['vision'] = self.vision_processor.get_capabilities()
        
        if self.audio_processor:
            capabilities['modalities']['audio'] = self.audio_processor.get_capabilities()
            
        if self.text_processor:
            capabilities['modalities']['text'] = self.text_processor.get_capabilities()
            
        if self.fusion_processor:
            capabilities['fusion'] = self.fusion_processor.get_capabilities()
        
        return capabilities
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics"""
        avg_processing_time = (
            sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            if self.metrics['processing_times'] else 0
        )
        
        return {
            'total_processed': self.metrics['total_processed'],
            'average_processing_time': avg_processing_time,
            'recent_processing_times': self.metrics['processing_times'][-10:],
            'perception_history_size': len(self.perception_history),
            'fusion_success_rate': self.metrics.get('fusion_success_rate', 0.0)
        }
    
    async def get_contextual_understanding(self, query: str) -> Dict[str, Any]:
        """
        Query the perception system for contextual understanding.
        
        Args:
            query: Natural language query about recent perceptions
            
        Returns:
            Dict containing relevant context and understanding
        """
        if not self.perception_history:
            return {'context': 'No recent perceptions available'}
        
        # Use recent perceptions to build context
        recent_perceptions = self.perception_history[-5:]
        
        context = {
            'recent_perceptions': len(recent_perceptions),
            'modalities_active': set(),
            'key_entities': [],
            'scene_elements': [],
            'temporal_sequence': []
        }
        
        for perception in recent_perceptions:
            # Extract modalities
            for input_id in perception.input_ids:
                if 'vision' in input_id:
                    context['modalities_active'].add('vision')
                elif 'audio' in input_id:
                    context['modalities_active'].add('audio')
                elif 'text' in input_id:
                    context['modalities_active'].add('text')
            
            # Extract scene understanding
            if 'objects' in perception.scene_understanding:
                context['scene_elements'].extend(perception.scene_understanding['objects'])
            
            # Build temporal sequence
            context['temporal_sequence'].append({
                'timestamp': perception.timestamp.isoformat(),
                'confidence': perception.overall_confidence,
                'elements': len(perception.cross_modal_relationships)
            })
        
        context['modalities_active'] = list(context['modalities_active'])
        
        # Simple query processing (can be enhanced with NLP)
        if 'recent' in query.lower() or 'last' in query.lower():
            context['focus'] = 'recent_activity'
        elif 'see' in query.lower() or 'visual' in query.lower():
            context['focus'] = 'vision'
        elif 'hear' in query.lower() or 'audio' in query.lower():
            context['focus'] = 'audio'
        elif 'understand' in query.lower() or 'meaning' in query.lower():
            context['focus'] = 'comprehension'
        
        return context
    
    async def shutdown(self):
        """Gracefully shutdown the perception engine"""
        logger.info("🔄 Shutting down Multi-Modal Perception Engine...")
        
        try:
            if self.fusion_processor:
                await self.fusion_processor.shutdown()
            if self.text_processor:
                await self.text_processor.shutdown()
            if self.audio_processor:
                await self.audio_processor.shutdown()
            if self.vision_processor:
                await self.vision_processor.shutdown()
            
            self.initialized = False
            logger.info("✅ Multi-Modal Perception Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

class PerceptionCapabilityTester:
    """Test and validate perception capabilities"""
    
    def __init__(self, engine: MultiModalPerceptionEngine):
        self.engine = engine
    
    async def test_vision_capabilities(self) -> Dict[str, Any]:
        """Test vision processing capabilities"""
        # Create a simple test image (red square)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 0, 0]  # Red square
        
        # Convert to bytes
        from PIL import Image
        img = Image.fromarray(test_image)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        try:
            result = await self.engine.perceive_vision(img_bytes, {'test': True})
            return {
                'status': 'success',
                'confidence': result.confidence,
                'features_detected': len(result.features),
                'entities_found': len(result.entities)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def test_audio_capabilities(self) -> Dict[str, Any]:
        """Test audio processing capabilities"""
        # Create a simple test audio (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Convert to bytes
        audio_bytes = audio_data.tobytes()
        
        try:
            result = await self.engine.perceive_audio(audio_bytes, {
                'sample_rate': sample_rate,
                'duration': duration,
                'test': True
            })
            return {
                'status': 'success',
                'confidence': result.confidence,
                'features_detected': len(result.features),
                'entities_found': len(result.entities)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def test_text_capabilities(self) -> Dict[str, Any]:
        """Test text processing capabilities"""
        test_text = """
        The Multi-Modal Perception Engine is a revolutionary system that enables
        artificial intelligence to understand and process multiple forms of sensory input
        simultaneously. This breakthrough technology combines computer vision, audio
        processing, and natural language understanding into a unified cognitive framework.
        """
        
        try:
            result = await self.engine.perceive_text(test_text, {'test': True})
            return {
                'status': 'success',
                'confidence': result.confidence,
                'features_detected': len(result.features),
                'entities_found': len(result.entities)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def test_multimodal_fusion(self) -> Dict[str, Any]:
        """Test cross-modal fusion capabilities"""
        try:
            # Create test inputs for all modalities
            vision_input = PerceptionInput(
                input_id="test_vision",
                timestamp=datetime.now(),
                source="test",
                modality="vision",
                data=b"fake_image_data",
                metadata={'test': True}
            )
            
            text_input = PerceptionInput(
                input_id="test_text",
                timestamp=datetime.now(),
                source="test",
                modality="text",
                data="This is a test of the multi-modal perception system.",
                metadata={'test': True}
            )
            
            # Test fusion
            result = await self.engine.perceive([vision_input, text_input])
            
            return {
                'status': 'success',
                'overall_confidence': result.overall_confidence,
                'modalities_fused': len(result.input_ids),
                'relationships_found': len(result.cross_modal_relationships),
                'scene_elements': len(result.scene_understanding)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# Factory function
def create_multimodal_engine(config: Optional[Dict[str, Any]] = None) -> MultiModalPerceptionEngine:
    """Create and return a configured multi-modal perception engine"""
    return MultiModalPerceptionEngine(config)