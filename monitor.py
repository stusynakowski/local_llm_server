#!/usr/bin/env python3
"""
Monitor GPU usage and server performance.
"""

import time
import psutil
import subprocess
from datetime import datetime

def get_gpu_info():
    """Get GPU memory usage and utilization"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(',')
        return {
            'utilization': float(gpu_util),
            'memory_used': float(mem_used),
            'memory_total': float(mem_total),
            'temperature': float(temp)
        }
    except Exception as e:
        return None

def get_system_info():
    """Get CPU and RAM usage"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'ram_used': psutil.virtual_memory().used / (1024**3),  # GB
        'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
        'ram_percent': psutil.virtual_memory().percent
    }

def monitor(interval=2):
    """Monitor system and GPU in real-time"""
    print("Local LLM Server Monitor")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get info
            gpu = get_gpu_info()
            sys_info = get_system_info()
            
            # Clear screen (optional)
            print("\033[2J\033[H", end="")
            
            # Display
            print(f"Local LLM Server Monitor - {timestamp}")
            print("=" * 70)
            
            if gpu:
                print(f"\n🎮 GPU (RTX 3090)")
                print(f"   Utilization: {gpu['utilization']:.1f}%")
                print(f"   Memory:      {gpu['memory_used']:.0f} MB / {gpu['memory_total']:.0f} MB "
                      f"({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
                print(f"   Temperature: {gpu['temperature']:.0f}°C")
            else:
                print("\n🎮 GPU: Not available")
            
            print(f"\n💻 System")
            print(f"   CPU:  {sys_info['cpu_percent']:.1f}%")
            print(f"   RAM:  {sys_info['ram_used']:.1f} GB / {sys_info['ram_total']:.1f} GB "
                  f"({sys_info['ram_percent']:.1f}%)")
            
            print("\n" + "=" * 70)
            print(f"Refreshing every {interval} seconds... (Ctrl+C to stop)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor()
