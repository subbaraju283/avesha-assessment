#!/usr/bin/env python3
"""
Setup script for NASA Query System environment variables.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up environment variables for the NASA Query System."""
    
    print("🚀 NASA Query System Environment Setup")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Found existing .env file")
        load_env_file(env_file)
    else:
        print("📝 Creating new .env file...")
        create_env_file()
    
    # Test LLM initialization
    print("\n🔧 Testing LLM initialization...")
    test_llm_init()

def load_env_file(env_file: Path):
    """Load environment variables from .env file."""
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
                print(f"   Loaded: {key}")

def create_env_file():
    """Create a new .env file with template values."""
    env_content = """# NASA Query System Environment Variables
# Replace these with your actual API keys

# OpenAI API Key (required for GPT-4)
OPENAI_API_KEY=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

# Anthropic API Key (optional, for Claude)
ANTHROPIC_API_KEY=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

# Other optional settings
DEBUG=false
LOG_LEVEL=INFO
"""
    
    with open(".env", 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("⚠️  Please edit .env file with your actual API keys")

def test_llm_init():
    """Test LLM initialization."""
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.models.llm_manager import LLMManager
        import yaml
        
        # Load config
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            print("❌ Config file not found")
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Config loaded successfully")
        
        # Try to initialize LLM Manager
        try:
            llm_manager = LLMManager(config)
            providers = llm_manager.get_available_providers()
            print(f"✅ LLM Manager initialized successfully")
            print(f"   Available providers: {providers}")
            
            # Test with a simple query
            import asyncio
            async def test_query():
                try:
                    result = await llm_manager.generate("Hello, this is a test.", provider=providers[0])
                    print(f"✅ Test query successful: {result[:50]}...")
                except Exception as e:
                    print(f"❌ Test query failed: {e}")
            
            asyncio.run(test_query())
            
        except Exception as e:
            print(f"❌ LLM Manager initialization failed: {e}")
            print("\n🔧 Troubleshooting tips:")
            print("   1. Make sure you have set your API keys in .env file")
            print("   2. Check that your API keys are valid")
            print("   3. Ensure you have the required packages installed")
            print("   4. Try running: pip install -r requirements.txt")
    
    except Exception as e:
        print(f"❌ Setup failed: {e}")

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'openai',
        'anthropic', 
        'langchain',
        'langchain-openai',
        'langchain-anthropic',
        'sentence-transformers',
        'click',
        'rich',
        'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("✅ All required packages are installed")

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    check_dependencies()
    
    print("\n🔧 Setting up environment...")
    setup_environment()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python main.py query 'Your question here'")
    print("3. Or run: python main.py interactive") 