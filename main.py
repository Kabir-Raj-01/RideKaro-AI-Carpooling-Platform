"""
main.py
═══════════════════════════════════════════════════════════════
Intelligent Carpooling Optimisation System – Entry Point
═══════════════════════════════════════════════════════════════
AI Lab Semester Final Project
  Kabir Raj   – 23K-0702
  Hassnain Aziz – 23K-0905
  NUCES (FAST), Karachi

Usage
-----
  python main.py                  # default scenario (hour=8, weather=clear)
  python main.py --hour 17        # evening rush
  python main.py --weather rainy  # rainy conditions
  python main.py --drivers 8 --passengers 20
  python main.py --no-plots       # skip matplotlib charts
  python main.py --test           # run unit tests and exit
"""
import argparse
import sys
import os

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.city_graph import build_karachi_demo_city
from models.bayesian import build_predictor
from data.data_generator import generate_scenario
from agents.agent import OrchestratorAgent
from utils.reporter import full_report, _section, INFO, RESET, HEADER, BOLD
from dashboard.visualiser import render_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Carpooling Optimisation System"
    )
    parser.add_argument("--hour",       type=int,   default=8,      help="Simulation hour (0-23)")
    parser.add_argument("--weather",    type=str,   default="clear",
                        choices=["clear", "rainy", "foggy"],         help="Weather condition")
    parser.add_argument("--drivers",    type=int,   default=6,      help="Number of drivers")
    parser.add_argument("--passengers", type=int,   default=15,     help="Number of passengers")
    parser.add_argument("--pop-size",   type=int,   default=60,     help="GA population size")
    parser.add_argument("--generations",type=int,   default=80,     help="GA generations")
    parser.add_argument("--seed",       type=int,   default=42,     help="Random seed")
    parser.add_argument("--no-plots",   action="store_true",        help="Skip matplotlib plots")
    parser.add_argument("--test",       action="store_true",        help="Run unit tests and exit")
    return parser.parse_args()


def run_tests():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)


def main():
    args = parse_args()

    if args.test:
        run_tests()

    print(HEADER + "\n  Initialising Intelligent Carpooling Optimisation System...\n" + RESET)

    # ── 1. Build city ──────────────────────────────────────────
    print(INFO + "  [1/4] Building Karachi city road network..." + RESET)
    city, locs = build_karachi_demo_city()
    print(f"        {city}")

    # ── 2. Train Bayesian model ────────────────────────────────
    print(INFO + "  [2/4] Training Bayesian demand predictor..." + RESET)
    predictor = build_predictor()
    print(f"        Predictor trained on 1000 synthetic records.")

    # ── 3. Generate scenario ───────────────────────────────────
    print(INFO + "  [3/4] Generating scenario "
          f"(hour={args.hour}, weather={args.weather}, "
          f"drivers={args.drivers}, passengers={args.passengers})..." + RESET)
    drivers, passengers = generate_scenario(
        city,
        n_drivers=args.drivers,
        n_passengers=args.passengers,
        hour=args.hour,
        seed=args.seed,
    )
    print(f"        {len(drivers)} drivers, {len(passengers)} passengers created.")

    # ── 4. Run multi-agent pipeline ────────────────────────────
    print(INFO + "  [4/4] Launching multi-agent pipeline...\n" + RESET)
    orchestrator = OrchestratorAgent(
        city=city,
        predictor=predictor,
        hour=args.hour,
        weather=args.weather,
    )
    # Override GA params from CLI
    orchestrator.ga_agent.pop_size = args.pop_size
    orchestrator.ga_agent.generations = args.generations

    report = orchestrator.run(drivers, passengers)

    # ── 5. Terminal report ─────────────────────────────────────
    full_report(report, ga_history=orchestrator.ga_agent.ga_history)

    # ── 6. Visualisations ──────────────────────────────────────
    if not args.no_plots:
        _section("GENERATING VISUALISATIONS")
        render_all(city, report, orchestrator.ga_agent.ga_history, predictor)
        print(INFO + "\n  All charts saved to ./output/" + RESET)

    return report


if __name__ == "__main__":
    main()
