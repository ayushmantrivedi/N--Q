"""
Nāḍī-Q test suite root.

Test tiers:
    unit/        — Fast, isolated tests. No I/O, no network, no simulators.
                   Run on every commit.
    integration/ — Tests requiring multiple components working together.
                   May use synthetic data. Run on every PR.
    scientific/  — Tests verifying mathematical and scientific correctness.
                   May be slower (circuit simulation). Run before any result claim.

Pytest markers:
    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.scientific
    @pytest.mark.slow
    @pytest.mark.requires_quantum
"""
