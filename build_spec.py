# build_spec.py - 使用spec文件解决命令行过长问题
import os
import sys
import subprocess
import importlib.util
import shutil


def install_dependencies():
    """安装必要的依赖"""
    required = ["pyinstaller", "pyaudio", "numpy", "funasr", "websocket-client", "requests"]

    print("检查并安装必要的依赖...")
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def locate_package_files(package_name):
    """定位包文件的路径"""
    try:
        package_spec = importlib.util.find_spec(package_name)
        if package_spec is None:
            return None

        package_path = os.path.dirname(package_spec.origin)
        print(f"找到包 {package_name} 位置: {package_path}")
        return package_path
    except Exception as e:
        print(f"查找包 {package_name} 时出错: {e}")
        return None


def create_spec_file():
    """创建PyInstaller spec文件"""
    # 定位FunASR包路径
    funasr_path = locate_package_files("funasr")
    if not funasr_path:
        print("错误: 无法定位FunASR包路径，请确保正确安装了FunASR")
        return False

    print(f"正在创建spec文件...")

    # 转换路径格式为Python字符串
    funasr_path_str = repr(funasr_path)

    # 创建spec文件内容
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 收集FunASR的所有数据文件
funasr_datas = []
funasr_path = {funasr_path_str}

# 手动添加version.txt
version_file = os.path.join(funasr_path, 'version.txt')
if os.path.exists(version_file):
    funasr_datas.append((version_file, 'funasr'))
else:
    print("警告: FunASR version.txt 不存在")

# 收集FunASR的所有数据文件
for root, dirs, files in os.walk(funasr_path):
    for file in files:
        file_path = os.path.join(root, file)
        rel_dir = os.path.relpath(os.path.dirname(file_path), os.path.dirname(funasr_path))
        funasr_datas.append((file_path, rel_dir))

# 添加项目模块
project_datas = [
    ('funasr_module.py', '.'),
    ('translation_module.py', '.'),
    ('realtime_tts.py', '.'),
    ('funasr_wrapper.py', '.'),
]

# 收集所有子模块
hidden_imports = [
    'pyaudio', 
    'numpy', 
    'websocket', 
    'funasr',
    'requests', 
    'httpx', 
    'ujson'
] + collect_submodules('funasr')

a = Analysis(
    ['main_translator.py'],
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
    name='实时语音翻译系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""

    with open("translator.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)

    print("Spec文件已创建: translator.spec")
    return True


def create_wrapper_module():
    """创建一个包装模块，以更安全的方式导入FunASR"""
    wrapper_content = """# funasr_wrapper.py - FunASR安全导入包装器
import os
import sys
import importlib

# 修复可能的路径问题
def fix_funasr_import():
    try:
        import funasr
        return True
    except (ImportError, FileNotFoundError) as e:
        print(f"尝试常规导入FunASR失败: {e}")

        # 尝试修复路径
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        funasr_dir = os.path.join(base_dir, 'funasr')

        if os.path.exists(funasr_dir) and funasr_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(funasr_dir))
            print(f"已添加FunASR路径: {os.path.dirname(funasr_dir)}")

            # 创建缺失的version.txt（如果需要）
            version_file = os.path.join(funasr_dir, 'version.txt')
            if not os.path.exists(version_file):
                try:
                    os.makedirs(os.path.dirname(version_file), exist_ok=True)
                    with open(version_file, 'w') as f:
                        f.write("1.0.0")
                    print(f"已创建缺失的version.txt文件: {version_file}")
                except Exception as ve:
                    print(f"创建version.txt失败: {ve}")

            try:
                import funasr
                return True
            except Exception as inner_e:
                print(f"修复路径后仍然无法导入FunASR: {inner_e}")
                return False
        else:
            print(f"未找到FunASR目录或已在路径中: {funasr_dir}")
            return False

# 安全导入AutoModel
def get_auto_model():
    if fix_funasr_import():
        try:
            from funasr import AutoModel
            return AutoModel
        except Exception as e:
            print(f"导入AutoModel失败: {e}")
            return None
    return None

# 预加载AutoModel
AutoModel = get_auto_model()
"""

    with open("funasr_wrapper.py", "w", encoding="utf-8") as f:
        f.write(wrapper_content)
    print("已创建FunASR安全导入包装器")


def modify_funasr_module():
    """修改funasr_module.py，使用安全导入方式"""
    # 读取原始文件
    with open("funasr_module.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 备份原始文件
    with open("funasr_module.py.bak", "w", encoding="utf-8") as f:
        f.write(content)

    # 替换导入语句
    modified_content = content.replace(
        "from funasr import AutoModel",
        "from funasr_wrapper import AutoModel"
    )

    # 写入修改后的文件
    with open("funasr_module.py", "w", encoding="utf-8") as f:
        f.write(modified_content)

    print("已修改funasr_module.py使用安全导入方式")


def restore_funasr_module():
    """恢复原始的funasr_module.py"""
    if os.path.exists("funasr_module.py.bak"):
        shutil.copy("funasr_module.py.bak", "funasr_module.py")
        os.remove("funasr_module.py.bak")
        print("已恢复原始funasr_module.py")


def build_with_spec():
    """使用spec文件构建应用"""
    print("\n开始使用spec文件构建应用...")

    # 使用spec文件构建
    cmd = ["pyinstaller", "--clean", "translator.spec"]
    subprocess.run(cmd)

    # 检查构建结果
    exe_path = os.path.join("dist", "实时语音翻译系统.exe")
    if os.path.exists(exe_path):
        print(f"\n构建成功！可执行文件已生成: {exe_path}")
        print("您可以将此文件复制到任何Windows电脑上直接运行!")
    else:
        print("\n构建似乎完成，但未找到EXE文件，请检查dist目录。")


def main():
    print("===== 实时语音翻译系统打包工具 (spec文件版) =====\n")

    # 安装依赖
    install_dependencies()

    # 创建FunASR包装器
    create_wrapper_module()

    # 修改funasr_module使用包装器
    modify_funasr_module()

    try:
        # 创建spec文件
        if create_spec_file():
            # 构建EXE
            build_with_spec()
    finally:
        # 恢复原始文件
        restore_funasr_module()

    input("\n按回车键退出...")


if __name__ == "__main__":
    main()