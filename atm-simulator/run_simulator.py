"""
Simple script to run the ATM simulator
"""

import asyncio

from simulator import ATMSimulator


async def main():
    print("🏪 ATM Telemetry Simulator")
    print("Press Ctrl+C to stop\n")

    simulator = ATMSimulator()

    try:
        await simulator.start_simulation()
    except KeyboardInterrupt:
        print("\n👋 Simulator stopped")


if __name__ == "__main__":
    asyncio.run(main())
