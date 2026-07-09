import subprocess
import sys
import os
import shutil
import platform

def run_command(command):
    """Runs a shell command and prints it."""
    print(f">>> Running: {command}")
    subprocess.check_call(command, shell=True)

def main():
    print("=" * 50)
    print("📦 RAG Project Auto-Setup Script")
    print("=" * 50)

    # 1. Create virtual environment
    venv_dir = ".venv"
    if not os.path.exists(venv_dir):
        print("\n1. Creating virtual environment (.venv)...")
        run_command(f"{sys.executable} -m venv {venv_dir}")
    else:
        print("\n1. Virtual environment already exists.")

    # 2. Determine the pip path inside the venv
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
        python_path = os.path.join(venv_dir, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")

    # 3. Upgrade pip and install requirements
    print("\n2. Installing dependencies from requirements.txt...")
    run_command(f"{pip_path} install --upgrade pip")
    run_command(f"{pip_path} install -r requirements.txt")

    # 4. Download the embedding model
    print("\n3. Downloading MiniLM embedding model (this may take a minute)...")
    if os.path.exists("scripts/download_model.py"):
        run_command(f"{python_path} scripts/download_model.py")
    else:
        print("⚠️  Warning: scripts/download_model.py not found. Please download the model manually.")

    # 5. Create .env from template
    if not os.path.exists(".env") and os.path.exists(".env.template"):
        print("\n4. Creating .env file from template...")
        shutil.copy(".env.template", ".env")
        print("✅ .env file created. Please open it and add your API keys!")

    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print(f"To activate your environment:")
    if platform.system() == "Windows":
        print(f"   .venv\\Scripts\\activate")
    else:
        print(f"   source .venv/bin/activate")
    print("To run the backend: uvicorn api.main:app --reload")
    print("To run the UI: streamlit run ui/app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()