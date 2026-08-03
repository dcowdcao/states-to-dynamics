"""Game corpora: stream records from HF and cut the prefixes every study branches from.

Imported by the rules, straightness, violation-divergence and bracket-flip pipelines —
they all start from the same legal-prefix corpus, and they must agree on it.

Importing this module also forces IPv4 and puts the Karvonen fork on the path; several
source-repo modules imported it for that side effect alone. Keep those effects here and
explicit rather than scattered, but do not make them lazy — HF downloads on the cluster
depend on force_ipv4 having run before the first request.
"""

# def stream_games(n, seed=None): ...        # HF othellogpt records -> move sequences
# def collect_legal(n, t_min, t_max): ...    # legal prefixes, round-robin over t
# def first_illegal_pos(moves): ...          # index of the first rule-violating move, or None
