"""Solver-agnostic modeling and analysis utilities.
"""
from .aggregation import TIME_COST_SCALE, compute_total_time_ast
from .analysis import analyze_constraints, ConstraintAnalysis, ConstraintStatus
from .domains import parse_user_block_sizes, derive_domains_from_etg, get_variant_name
from .memory_access import summarize_memory_accesses, walk_workloads
from .reporter import print_breakdown, print_active_constraints, print_unsat_core, print_mus, print_result_summary
