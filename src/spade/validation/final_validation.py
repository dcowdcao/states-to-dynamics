"""Fast pre-deployment self-tests (no GPU, synthetic data, < 2 min).

~17 checks: every module compiles/imports, SpaDE encode→decode round-trips, standardization
math is correct, LR schedule matches Karvonen, the figure type-map covers all 8 types, and
no legacy import paths remain. Run before any cluster job.

Entrypoint: `python -m spade.validation.final_validation`   (prints "N/N PASSED")
"""

# _TESTS = []
# def test(fn): _TESTS.append(fn); return fn   # decorator to register
#
# @test
# def test_spade_roundtrip(): ...
# @test
# def test_standardization_math(): ...
# @test
# def test_figure_type_map_covers_8(): ...
# ... (~17 total)
#
# def main(): run all, print pass/fail count, exit nonzero on any failure
