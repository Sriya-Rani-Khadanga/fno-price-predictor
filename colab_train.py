"""
Colab Orchestrator — Starts all 5 MCP servers and launches training.
Use this script in a Google Colab cell to automate the setup.
"""
import subprocess
import time
import requests
import sys
import os

SERVERS = [
    {"name": "market_data", "path": "mcp_servers/market_data_server.py", "port": 8001},
    {"name": "risk", "path": "mcp_servers/risk_server.py", "port": 8002},
    {"name": "cost", "path": "mcp_servers/cost_server.py", "port": 8003},
    {"name": "technical", "path": "mcp_servers/technical_server.py", "port": 8004},
    {"name": "news", "path": "mcp_servers/news_server.py", "port": 8005},
]

def start_servers():
    processes = []
    print("🚀 Starting 5 MCP Brain Servers...")
    for server in SERVERS:
        print(f"  -> Launching {server['name']} on port {server['port']}...")
        proc = subprocess.Popen([sys.executable, server['path']], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT)
        processes.append(proc)
    
    # Wait for health checks
    print("\n⏳ Waiting for servers to initialize (Health Check)...")
    for _ in range(15):
        all_healthy = True
        for server in SERVERS:
            try:
                resp = requests.get(f"http://localhost:{server['port']}/health", timeout=1)
                if resp.status_code != 200: all_healthy = False
            except:
                all_healthy = False
        
        if all_healthy:
            print("✅ All servers are healthy and connected!")
            return processes
        time.sleep(2)
    
    print("⚠️ Some servers failed to start in time. Check logs.")
    return processes

def run_training(steps=3000):
    print(f"\n🧠 Starting Multi-Factor GRPO Training for {steps} steps...")
    try:
        # Run main.py in training mode
        subprocess.run([sys.executable, "main.py", "--mode", "train", "--steps", str(steps)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    server_procs = start_servers()
    try:
        run_training(steps=3000)
    finally:
        print("\n🧹 Cleaning up servers...")
        for p in server_procs:
            p.terminate()
        print("Done.")
