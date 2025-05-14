#!/usr/bin/env python
# improved_build_script.py - Improved packaging tool
import os
import sys
import subprocess
import importlib.util
import shutil
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def print_section(title: str):
    """Print section title with separators"""
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def install_dependencies():
    """Install required dependencies"""
    print_section("Installing Dependencies")

    required = [
        "pyinstaller",
        "pyaudio",
        "numpy",
        "funasr",
        "websocket-client",
        "requests",
        "httpx",
        "ujson"
    ]

    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def locate_package_files(package_name: str) -> Optional[str]:
    """Locate package files path"""
    try:
        package_spec = importlib.util.find_spec(package_name)
        if package_spec is None:
            return None

        package_path = os.path.dirname(package_spec.origin)
        print(f"Found package {package_name} at: {package_path}")
        return package_path
    except Exception as e:
        print(f"Error locating package {package_name}: {e}")
        return None


def locate_all_modules() -> Dict[str, str]:
    """Locate all required module paths"""
    modules = {
        "funasr": None,
        "websocket": None,
        "httpx": None,
        "ujson": None,
        "requests": None,
        "numpy": None,
    }

    for module_name in modules.keys():
        modules[module_name] = locate_package_files(module_name)

    # Print all found modules
    print_section("Found Python Modules")
    for name, path in modules.items():
        status = "✓" if path else "✗"
        print(f"{status} {name}: {path or 'Not found'}")

    return modules


def create_debug_module():
    """Create debug module for runtime information"""
    print_section("Creating Debug Module")

    debug_content = """# debug_helper.py - Debug helper module
import os
import sys
import platform
import json
import traceback

def get_runtime_info():

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cwd": os.getcwd(),
        "exe_path": sys.executable,
        "path_env": os.environ.get("PATH", ""),
        "sys_path": sys.path,
        "bundled_files": []
    }

    # Get list of bundled files
    if hasattr(sys, "_MEIPASS"):
        base_dir = sys._MEIPASS
        info["base_dir"] = base_dir

        # List files in base directory
        all_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                all_files.append(rel_path)

        info["bundled_files"] = all_files

    return info

def save_runtime_info(filename="runtime_info.json"):

    try:
        info = get_runtime_info()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"Runtime information saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving runtime information: {e}")
        traceback.print_exc()
        return False

def check_network(host="itrans.xf-yun.cn", port=443):

    import socket

    try:
        socket.create_connection((host, port), timeout=5)
        return True
    except Exception as e:
        print(f"Network connection error: {e}")
        return False

def monitor_module(module_name, function_name):

    # Save original function
    module = __import__(module_name)
    original_function = getattr(module, function_name)

    def wrapper(*args, **kwargs):
        print(f"Calling {module_name}.{function_name} - Args: {args}, {kwargs}")
        try:
            result = original_function(*args, **kwargs)
            print(f"{module_name}.{function_name} executed successfully")
            return result
        except Exception as e:
            print(f"{module_name}.{function_name} execution failed: {e}")
            traceback.print_exc()
            raise

    # Replace original function
    setattr(module, function_name, wrapper)
    print(f"Monitoring {module_name}.{function_name}")

# Record environment information at startup
save_runtime_info()
"""

    with open("debug_helper.py", "w", encoding="utf-8") as f:
        f.write(debug_content)

    print("Debug helper module created: debug_helper.py")


def create_network_patch_module():
    """Create network patch module"""
    print_section("Creating Network Patch Module")

    network_patch_content = """# network_patch.py - Fix network access issues in packaged app
import os
import sys
import ssl
import socket
import urllib.request
import certifi

def apply_network_patches():
 
    # Set SSL certificate path
    try:
        # Try using certifi package
        os.environ['SSL_CERT_FILE'] = certifi.where()
        print(f"SSL certificate path set to: {os.environ['SSL_CERT_FILE']}")
    except Exception as e:
        print(f"Failed to set SSL certificate path: {e}")

    # Disable SSL verification (may be needed in some environments)
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        print("SSL verification disabled")
    except Exception as e:
        print(f"Failed to disable SSL verification: {e}")

    # Ensure reasonable timeout settings
    socket.setdefaulttimeout(30)  # 30 second timeout

    # Test network connections
    test_network_connection()

def test_network_connection():

    test_urls = [
        "https://www.baidu.com",
        "https://itrans.xf-yun.cn/",
        "https://tts-api.xfyun.cn/"
    ]

    for url in test_urls:
        try:
            print(f"Testing connection to {url}...")
            response = urllib.request.urlopen(url, timeout=5)
            print(f"Connection successful: {response.status} {response.reason}")
        except Exception as e:
            print(f"Connection failed: {e}")

if __name__ == "__main__":
    apply_network_patches()
"""

    with open("network_patch.py", "w", encoding="utf-8") as f:
        f.write(network_patch_content)

    print("Network patch module created: network_patch.py")


def create_tts_patch_module():
    """Create TTS patch module"""
    print_section("Creating TTS Patch Module")

    tts_patch_content = """# tts_patch.py - Fix TTS functionality in packaged app
import os
import sys
import time
import threading
import traceback
import gc

def apply_tts_patches():

    try:
        # Fix WebSocket connection issues
        patch_websocket()

        # Fix SSL certificate issues
        patch_ssl_certs()

        # Fix TTS playback status tracking
        patch_tts_status_tracking()

        print("TTS fixes applied")
    except Exception as e:
        print(f"Error applying TTS fixes: {e}")
        traceback.print_exc()

def patch_websocket():

    try:
        import websocket
        original_create_connection = websocket.create_connection

        def patched_create_connection(*args, **kwargs):
            print(f"WebSocket connection request: {args}, {kwargs}")
            # Add SSL-related options
            if "sslopt" not in kwargs:
                kwargs["sslopt"] = {"cert_reqs": 0}  # 0 = CERT_NONE
            try:
                conn = original_create_connection(*args, **kwargs)
                print("WebSocket connection successful")
                return conn
            except Exception as e:
                print(f"WebSocket connection failed: {e}")
                raise

        websocket.create_connection = patched_create_connection
        print("WebSocket create_connection function patched")
    except Exception as e:
        print(f"Error patching WebSocket connection: {e}")

def patch_ssl_certs():

    try:
        import ssl

        # Backup original context creation function
        original_context = ssl._create_default_https_context

        # Use appropriate certificate verification based on situation
        def flexible_context():
            try:
                return original_context()
            except Exception:
                print("SSL certificate verification failed, using unverified context")
                return ssl._create_unverified_context()

        # Replace with flexible context creation function
        ssl._create_default_https_context = flexible_context
        print("SSL certificate verification function patched")
    except Exception as e:
        print(f"Error patching SSL certificate verification: {e}")

def patch_tts_status_tracking():

    # Check if using realtime_tts module
    try:
        import realtime_tts

        # Ensure TTS status reset guardian function
        def tts_status_watchdog():


            while True:
                try:
                    # Check all threads
                    for thread in threading.enumerate():
                        if thread.name.startswith("tts_") and not thread.is_alive():
                            print(f"Detected stopped TTS thread: {thread.name}")
                            # Try to find and reset TTS instance
                            import main_translator
                            for obj in gc.get_objects():
                                if isinstance(obj, main_translator.RealtimeTranslator):
                                    print("Resetting translator status")
                                    obj.is_speaking = False
                except Exception as e:
                    print(f"TTS status monitoring error: {e}")

                time.sleep(5)  # Check every 5 seconds

        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=tts_status_watchdog, 
            daemon=True,
            name="tts_status_monitor"
        )
        monitor_thread.start()
        print("TTS status monitoring thread started")

    except ImportError:
        print("realtime_tts module not found, skipping TTS status tracking fix")
    except Exception as e:
        print(f"Error patching TTS status tracking: {e}")

if __name__ == "__main__":
    apply_tts_patches()
"""

    with open("tts_patch.py", "w", encoding="utf-8") as f:
        f.write(tts_patch_content)

    print("TTS patch module created: tts_patch.py")


def create_main_launcher():
    """Create launcher script"""
    print_section("Creating Launcher Script")

    launcher_content = """# main_launcher.py - Application launcher
import os
import sys
import importlib
import traceback

def apply_patches():

    try:
        # 1. Import and apply debug helper
        print("Importing debug helper...")
        import debug_helper

        # 2. Import and apply network fixes
        print("Applying network fixes...")
        import network_patch
        network_patch.apply_network_patches()

        # 3. Import and apply TTS fixes
        print("Applying TTS fixes...")
        import tts_patch
        tts_patch.apply_tts_patches()

        print("All fixes applied")
        return True
    except Exception as e:
        print(f"Error applying fixes: {e}")
        traceback.print_exc()
        return False

def run_main_app():
  
    try:
        # Import and run main program
        print("Starting real-time speech translation system...")
        import main_translator

        # API keys configuration
        # Machine translation and speech synthesis API keys
        APP_ID = "86c79fb7"
        API_KEY = "f4369644e37eddd43adfe436e7904cf1"
        API_SECRET = "MDY3ZGFkYWEyZDBiOTJkOGIyOTllOWMz"

        # Real-time speech transcription API keys
        ASR_APP_ID = "86c79fb7"
        ASR_API_KEY = "acf74303ddb1af7196de01aadd232feb"

        # Create and run translation system
        translator = main_translator.RealtimeTranslator(ASR_APP_ID, ASR_API_KEY, APP_ID, API_KEY, API_SECRET)
        translator.run()

    except Exception as e:
        print(f"Error running main program: {e}")
        traceback.print_exc()
        input("\\nPress Enter to exit...")

if __name__ == "__main__":
    # Apply fixes
    if apply_patches():
        # Run main program
        run_main_app()
    else:
        print("Cannot apply necessary fixes, program cannot start")
        input("\\nPress Enter to exit...")
"""

    with open("main_launcher.py", "w", encoding="utf-8") as f:
        f.write(launcher_content)

    print("Launcher script created: main_launcher.py")


def create_improved_spec_file(modules: Dict[str, str]):
    """Create improved PyInstaller spec file"""
    print_section("Creating Improved Spec File")

    # Ensure funasr path is available
    funasr_path = modules.get("funasr")
    if not funasr_path:
        print("Error: Cannot locate FunASR package path, please ensure FunASR is correctly installed")
        return False

    # Handle websocket library
    websocket_path = modules.get("websocket")
    if not websocket_path:
        print("Warning: websocket package not found, this may cause TTS functionality issues")

    # Convert paths to Python string format
    funasr_path_str = repr(funasr_path)
    websocket_path_str = repr(websocket_path) if websocket_path else "None"

    print(f"Creating improved spec file...")

    # Create spec file content
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Key paths
funasr_path = {funasr_path_str}
websocket_path = {websocket_path_str}

# Collect FunASR data files
funasr_datas = []

# Manually add version.txt
version_file = os.path.join(funasr_path, 'version.txt')
if os.path.exists(version_file):
    funasr_datas.append((version_file, 'funasr'))
else:
    print("Warning: FunASR version.txt does not exist")
    # Try to create a virtual version.txt
    try:
        with open(os.path.join(funasr_path, 'version.txt'), 'w') as f:
            f.write("1.0.0")
        print("Created temporary version file")
        funasr_datas.append((os.path.join(funasr_path, 'version.txt'), 'funasr'))
    except Exception as e:
        print(f"Failed to create temporary version file: {{e}}")

# Collect all FunASR data files
for root, dirs, files in os.walk(funasr_path):
    for file in files:
        file_path = os.path.join(root, file)
        rel_dir = os.path.relpath(os.path.dirname(file_path), os.path.dirname(funasr_path))
        funasr_datas.append((file_path, rel_dir))

# Add project modules
project_datas = [
    ('funasr_module.py', '.'),
    ('translation_module.py', '.'),
    ('realtime_tts.py', '.'),
    ('main_translator.py', '.'),
    ('debug_helper.py', '.'),
    ('network_patch.py', '.'),
    ('tts_patch.py', '.'),
]

# Add certificate files
try:
    import certifi
    cert_file = certifi.where()
    if os.path.exists(cert_file):
        project_datas.append((cert_file, '.'))
        print(f"Added certificate file: {{cert_file}}")
except ImportError:
    print("certifi package not installed, cannot add SSL certificates")

# Collect all submodules
hidden_imports = [
    'pyaudio', 
    'numpy', 
    'websocket', 
    'websocket._core',
    'websocket._app',
    'websocket._abnf',
    'websocket._exceptions',
    'websocket._handshake',
    'websocket._http',
    'websocket._logging',
    'websocket._socket',
    'websocket._ssl_compat',
    'websocket._url',
    'websocket._utils',
    'funasr',
    'requests', 
    'httpx', 
    'ujson',
    'ssl',
    'certifi',
    'debug_helper',
    'network_patch',
    'tts_patch',
    '_thread',
    'importlib',
    'importlib.util',
    'typing',
    'json',
    'urllib',
    'threading',
    'queue',
] + collect_submodules('funasr')

a = Analysis(
    ['main_launcher.py'],  # Use launcher as entry point
    pathex=[],
    binaries=[],
    datas=project_datas + funasr_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RealTimeTranslator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True to view error output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""

    with open("improved_translator.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)

    print("Improved spec file created: improved_translator.spec")
    return True


def build_with_improved_spec():
    """Build with improved spec file"""
    print_section("Building Application")

    # Build using spec file
    cmd = ["pyinstaller", "--clean", "improved_translator.spec"]
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

    # Check build result
    exe_path = os.path.join("dist", "RealTimeTranslator.exe")
    if os.path.exists(exe_path):
        print(f"\nBuild successful! Executable generated: {exe_path}")
        print("You can copy this file to any Windows computer and run it directly!")
        return True
    else:
        print("\nBuild appears complete, but EXE file not found. Check the dist directory.")
        return False


def main():
    print("=" * 70)
    print("= REAL-TIME TRANSLATION SYSTEM PACKAGING TOOL (IMPROVED VERSION) =")
    print("=" * 70)
    print("\nThis tool will create multiple helper modules to fix incomplete functionality issues")
    print("Main issues addressed:")
    print("1. Network connection and SSL certificate issues")
    print("2. WebSocket connection failure issues")
    print("3. Resource file path issues")
    print("4. Status tracking and recovery issues")
    print("\nStarting packaging process...\n")

    try:
        # Step 1: Install dependencies
        install_dependencies()

        # Step 2: Locate all modules
        modules = locate_all_modules()

        # Step 3: Create debug helper module
        create_debug_module()

        # Step 4: Create network patch module
        create_network_patch_module()

        # Step 5: Create TTS patch module
        create_tts_patch_module()

        # Step 6: Create launcher script
        create_main_launcher()

        # Step 7: Create improved spec file
        if create_improved_spec_file(modules):
            # Step 8: Build with improved spec file
            success = build_with_improved_spec()

            if success:
                print("\nBuild and packaging process successfully completed!")
                print("This version should fix the issue of translation and TTS not executing after speech recognition")
                print("\nIf problems persist, check the generated runtime_info.json file for debugging information")
            else:
                print("\nThere may be issues with the build process, please check error messages")

    except Exception as e:
        print(f"\nError in packaging process: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()