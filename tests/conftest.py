# AI Summary: Pytest configuration forcing a non‑interactive backend so
# matplotlib/corner plots can be created in CI without a display server.
import matplotlib

matplotlib.use("Agg")
