"""
Simple script to run the ATM simulator
"""
import asyncio
from simulator import ATMSimulator

if __name__ == "__main__":
    print("🏪 ATM Telemetry Simulator")
    print("Press Ctrl+C to stop\n")
    
    simulator = ATMSimulator()
    
    try:
        asyncio.run(simulator.start_simulation())
    except KeyboardInterrupt:
        print("\n👋 Simulator stopped")
