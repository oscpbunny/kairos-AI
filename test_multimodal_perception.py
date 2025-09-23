"""
Project Kairos: Multi-Modal Perception Engine Demonstration
Phase 7 - Advanced Intelligence

Demonstrates the revolutionary multi-modal perception capabilities where
Kairos can simultaneously see, hear, and understand text with cross-modal fusion.
"""

import asyncio
import logging
import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import our multi-modal perception engine
from agents.enhanced.perception.multimodal_engine import (
    MultiModalPerceptionEngine, 
    PerceptionInput,
    PerceptionCapabilityTester,
    create_multimodal_engine
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kairos.multimodal.demo")

async def demonstrate_multimodal_perception():
    """Demonstrate the revolutionary multi-modal perception capabilities"""
    
    print("\n" + "="*80)
    print("🧠✨ PROJECT KAIROS - PHASE 7 ADVANCED INTELLIGENCE ✨🧠")
    print("Multi-Modal Perception Engine Demonstration")
    print("="*80)
    print()
    
    # Create and initialize the multi-modal perception engine
    print("🚀 Initializing Multi-Modal Perception Engine...")
    config = {
        'vision': {
            'max_image_size': (1920, 1080),
            'supported_formats': ['PNG', 'JPEG', 'JPG', 'BMP']
        },
        'audio': {
            'sample_rate': 16000,
            'max_duration_seconds': 300
        },
        'text': {
            'max_length': 10000
        },
        'fusion': {
            'attention_mechanism': 'cross_modal',
            'confidence_threshold': 0.5
        }
    }
    
    engine = create_multimodal_engine(config)
    
    # Initialize the engine
    success = await engine.initialize()
    if not success:
        print("❌ Failed to initialize Multi-Modal Perception Engine")
        return
    
    print("✅ Multi-Modal Perception Engine initialized successfully!")
    print()
    
    # Display capabilities
    capabilities = engine.get_capabilities()
    print("🎯 PERCEPTION CAPABILITIES:")
    print(f"   Version: {capabilities['version']}")
    print(f"   Modalities: {list(capabilities['modalities'].keys())}")
    if 'fusion' in capabilities:
        fusion_caps = capabilities['fusion']['capabilities']
        print(f"   Fusion: {', '.join(fusion_caps)}")
    print()
    
    # Demonstrate individual modality processing
    await demonstrate_vision_processing(engine)
    await demonstrate_audio_processing(engine)
    await demonstrate_text_processing(engine)
    
    # Demonstrate the revolutionary multi-modal fusion
    await demonstrate_multimodal_fusion(engine)
    
    # Demonstrate contextual understanding
    await demonstrate_contextual_understanding(engine)
    
    # Performance metrics
    await display_performance_metrics(engine)
    
    # Capability testing
    await test_perception_capabilities(engine)
    
    # Cleanup
    print("🔄 Shutting down Multi-Modal Perception Engine...")
    await engine.shutdown()
    print("✅ Shutdown complete!")
    print()
    
    print("🎉 MULTI-MODAL PERCEPTION DEMONSTRATION COMPLETE!")
    print("   Kairos can now see, hear, and understand with unified intelligence.")
    print("   The future of AI perception is here! 🌟")
    print("="*80)

async def demonstrate_vision_processing(engine):
    """Demonstrate vision processing capabilities"""
    print("👁️ VISION PROCESSING DEMONSTRATION")
    print("-" * 40)
    
    try:
        # Create mock image data (in production, this would be actual image bytes)
        mock_image_data = b"mock_image_data_representing_a_scene_with_people_and_objects"
        
        # Process vision input
        result = await engine.perceive_vision(mock_image_data, {
            'source': 'demo_camera',
            'format': 'PNG',
            'resolution': '1920x1080'
        })
        
        print(f"   ✅ Image processed successfully")
        print(f"   🎯 Confidence: {result.confidence:.3f}")
        print(f"   🔍 Entities detected: {len(result.entities)}")
        
        # Show detected entities
        for i, entity in enumerate(result.entities[:3]):  # Show first 3
            print(f"      • {entity['description']} (confidence: {entity['confidence']:.2f})")
        
        if len(result.entities) > 3:
            print(f"      • ... and {len(result.entities) - 3} more entities")
        
        # Show scene analysis
        if hasattr(result, 'features'):
            scene_analysis = result.features.get('scene_analysis', {})
            scene_type = scene_analysis.get('scene_type', 'unknown')
            lighting = scene_analysis.get('lighting', 'unknown')
            print(f"   🎨 Scene: {scene_type} with {lighting} lighting")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Vision processing failed: {e}")
        print()

async def demonstrate_audio_processing(engine):
    """Demonstrate audio processing capabilities"""
    print("🎧 AUDIO PROCESSING DEMONSTRATION")
    print("-" * 40)
    
    try:
        # Create mock audio data
        mock_audio_data = b"mock_audio_data_with_speech_and_background_sounds"
        
        # Process audio input
        result = await engine.perceive_audio(mock_audio_data, {
            'sample_rate': 16000,
            'duration': 10.5,
            'format': 'WAV'
        })
        
        print(f"   ✅ Audio processed successfully")
        print(f"   🎯 Confidence: {result.confidence:.3f}")
        print(f"   🔊 Entities detected: {len(result.entities)}")
        
        # Show detected audio entities
        for entity in result.entities:
            entity_type = entity.get('type', 'unknown')
            description = entity.get('description', 'audio entity')
            transcription = entity.get('transcription')
            
            if transcription:
                print(f"      • {description}: '{transcription}'")
            else:
                print(f"      • {description}")
        
        # Show audio analysis
        if hasattr(result, 'features'):
            audio_features = result.features
            if 'speech_recognition' in audio_features:
                speech_info = audio_features['speech_recognition']
                word_count = speech_info.get('word_count', 0)
                if word_count > 0:
                    print(f"   🗣️ Speech detected: {word_count} words transcribed")
            
            if 'acoustic_scene' in audio_features:
                scene_info = audio_features['acoustic_scene']
                scene_type = scene_info.get('scene_type', 'unknown')
                print(f"   🏞️ Acoustic scene: {scene_type}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Audio processing failed: {e}")
        print()

async def demonstrate_text_processing(engine):
    """Demonstrate text processing capabilities"""
    print("📝 TEXT PROCESSING DEMONSTRATION")
    print("-" * 40)
    
    try:
        # Sample text for analysis
        sample_text = """
        Project Kairos represents a revolutionary breakthrough in artificial intelligence.
        Our multi-modal perception engine combines computer vision, natural language processing,
        and audio analysis to create truly intelligent autonomous systems. This technology
        enables AI agents to see, hear, and understand the world with unprecedented sophistication.
        Dr. Sarah Chen, the lead researcher, says this is "the future of AI perception."
        Contact us at info@kairos-ai.com for more information.
        """
        
        # Process text input
        result = await engine.perceive_text(sample_text, {
            'source': 'demo_document',
            'language': 'en'
        })
        
        print(f"   ✅ Text processed successfully")
        print(f"   🎯 Confidence: {result.confidence:.3f}")
        print(f"   📊 Entities detected: {len(result.entities)}")
        
        # Show detected text entities
        entity_types = {}
        for entity in result.entities:
            entity_type = entity.get('type', 'unknown')
            entity_text = entity.get('text', '')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity_text)
        
        for entity_type, texts in entity_types.items():
            print(f"      • {entity_type}: {', '.join(texts[:3])}")
        
        # Show text analysis
        if hasattr(result, 'features'):
            text_features = result.features
            if 'sentiment_analysis' in text_features:
                sentiment_info = text_features['sentiment_analysis']
                sentiment = sentiment_info.get('sentiment', 'neutral')
                score = sentiment_info.get('score', 0.0)
                print(f"   😊 Sentiment: {sentiment} (score: {score:.2f})")
            
            if 'topic_modeling' in text_features:
                topic_info = text_features['topic_modeling']
                primary_topic = topic_info.get('primary_topic', 'general')
                print(f"   📚 Primary topic: {primary_topic}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Text processing failed: {e}")
        print()

async def demonstrate_multimodal_fusion(engine):
    """Demonstrate the revolutionary cross-modal fusion"""
    print("🔗 MULTI-MODAL FUSION DEMONSTRATION")
    print("   ** This is where the magic happens! **")
    print("-" * 50)
    
    try:
        # Create multi-modal inputs
        vision_input = PerceptionInput(
            input_id="demo_vision",
            timestamp=datetime.now(),
            source="demo_camera",
            modality="vision",
            data=b"mock_image_showing_person_speaking_at_podium_with_kairos_logo",
            metadata={'resolution': '1920x1080', 'format': 'PNG'}
        )
        
        audio_input = PerceptionInput(
            input_id="demo_audio",
            timestamp=datetime.now(),
            source="demo_microphone",
            modality="audio",
            data=b"mock_audio_of_person_presenting_kairos_technology",
            metadata={'sample_rate': 16000, 'duration': 15.0}
        )
        
        text_input = PerceptionInput(
            input_id="demo_text",
            timestamp=datetime.now(),
            source="demo_transcript",
            modality="text",
            data="Welcome to the Kairos presentation. Today we'll demonstrate our revolutionary multi-modal AI system that can see, hear, and understand simultaneously.",
            metadata={'language': 'en'}
        )
        
        # Perform multi-modal fusion
        print("   🔄 Fusing vision, audio, and text inputs...")
        fused_result = await engine.perceive([vision_input, audio_input, text_input])
        
        print(f"   ✅ Multi-modal fusion completed!")
        print(f"   🎯 Overall confidence: {fused_result.overall_confidence:.3f}")
        print(f"   🧩 Modalities fused: {len(fused_result.input_ids)}")
        
        # Show unified understanding
        scene = fused_result.scene_understanding
        print(f"   🏞️ Scene type: {scene.get('scene_type', 'unknown')}")
        print(f"   🏢 Environment: {scene.get('environment', 'unknown')}")
        print(f"   🎯 Activity: {scene.get('activity', 'unknown')}")
        
        # Show cross-modal relationships
        relationships = fused_result.cross_modal_relationships
        print(f"   🔗 Cross-modal relationships found: {len(relationships)}")
        for i, rel in enumerate(relationships[:2]):  # Show first 2
            rel_type = rel.get('type', 'unknown')
            confidence = rel.get('confidence', 0.0)
            print(f"      • {rel_type} (confidence: {confidence:.2f})")
        
        # Show attention weights
        attention = fused_result.attention_map
        print("   🎯 Attention distribution:")
        for modality, weights in attention.items():
            global_attention = weights.get('global_attention', 0.0)
            print(f"      • {modality}: {global_attention:.2f}")
        
        # Show unified features
        unified = fused_result.unified_features
        print(f"   📊 Total entities across modalities: {unified.get('entity_count', 0)}")
        dominant = unified.get('dominant_modality', 'none')
        print(f"   🏆 Dominant modality: {dominant}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Multi-modal fusion failed: {e}")
        print()

async def demonstrate_contextual_understanding(engine):
    """Demonstrate contextual understanding capabilities"""
    print("🤔 CONTEXTUAL UNDERSTANDING DEMONSTRATION")
    print("-" * 45)
    
    try:
        # Test contextual queries
        queries = [
            "What did I see recently?",
            "What sounds were detected?", 
            "Can you understand the overall scene?",
            "What is the main activity happening?"
        ]
        
        for query in queries:
            context = await engine.get_contextual_understanding(query)
            print(f"   ❓ Query: {query}")
            
            if 'focus' in context:
                print(f"      Focus: {context['focus']}")
            
            if 'recent_perceptions' in context:
                count = context['recent_perceptions']
                print(f"      Recent perceptions: {count}")
            
            if 'modalities_active' in context:
                modalities = context['modalities_active']
                if modalities:
                    print(f"      Active modalities: {', '.join(modalities)}")
            
            print()
    
    except Exception as e:
        print(f"   ❌ Contextual understanding failed: {e}")
        print()

async def display_performance_metrics(engine):
    """Display performance metrics"""
    print("📊 PERFORMANCE METRICS")
    print("-" * 25)
    
    try:
        metrics = engine.get_metrics()
        print(f"   Total processed: {metrics['total_processed']}")
        print(f"   Average processing time: {metrics['average_processing_time']:.3f}s")
        print(f"   Perception history size: {metrics['perception_history_size']}")
        print(f"   Fusion success rate: {metrics['fusion_success_rate']:.2%}")
        print()
        
    except Exception as e:
        print(f"   ❌ Could not retrieve metrics: {e}")
        print()

async def test_perception_capabilities(engine):
    """Test the perception capabilities"""
    print("🧪 CAPABILITY TESTING")
    print("-" * 20)
    
    try:
        tester = PerceptionCapabilityTester(engine)
        
        # Test vision capabilities
        print("   🔍 Testing vision capabilities...")
        vision_test = await tester.test_vision_capabilities()
        status = "✅" if vision_test['status'] == 'success' else "❌"
        print(f"      {status} Vision test: {vision_test['status']}")
        
        # Test audio capabilities
        print("   🔊 Testing audio capabilities...")
        audio_test = await tester.test_audio_capabilities()
        status = "✅" if audio_test['status'] == 'success' else "❌"
        print(f"      {status} Audio test: {audio_test['status']}")
        
        # Test text capabilities
        print("   📝 Testing text capabilities...")
        text_test = await tester.test_text_capabilities()
        status = "✅" if text_test['status'] == 'success' else "❌"
        print(f"      {status} Text test: {text_test['status']}")
        
        # Test multi-modal fusion
        print("   🔗 Testing multi-modal fusion...")
        fusion_test = await tester.test_multimodal_fusion()
        status = "✅" if fusion_test['status'] == 'success' else "❌"
        print(f"      {status} Fusion test: {fusion_test['status']}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Capability testing failed: {e}")
        print()

def print_phase_7_banner():
    """Print the Phase 7 banner"""
    print("\n" + "🧠" * 20)
    print("🌟 KAIROS PHASE 7 - ADVANCED INTELLIGENCE 🌟")
    print("The birth of truly multi-modal artificial perception")
    print("🧠" * 20)
    print()
    print("Revolutionary Capabilities Achieved:")
    print("• 👁️  Computer Vision - Object detection, scene understanding")
    print("• 🎧  Audio Processing - Speech recognition, sound analysis")  
    print("• 📝  Text Analysis - NLP, sentiment analysis, entity extraction")
    print("• 🔗  Cross-Modal Fusion - Unified understanding across modalities")
    print("• 🧠  Attention Mechanisms - Intelligent focus allocation")
    print("• 🎯  Contextual Reasoning - Understanding meaning and relationships")
    print()

if __name__ == "__main__":
    print_phase_7_banner()
    
    try:
        asyncio.run(demonstrate_multimodal_perception())
    except KeyboardInterrupt:
        print("\n⚡ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()