# pattern: Imperative Shell

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_document(relative_path: str) -> str:
    path = _REPO_ROOT / relative_path
    assert path.is_file(), f"missing required documentation page: {relative_path}"
    return path.read_text()


def _normalized(relative_path: str) -> str:
    return " ".join(_read_document(relative_path).split())


def test_jax_api_docs_define_the_explicit_operator_and_private_candidate_contract() -> None:
    docs = _normalized("docs/api/jax.md")

    assert "lineararg_matmat(operator, values)" in docs
    assert "loss(parameters, operator, phenotype)" in docs
    assert "explicit argument" in docs
    assert "compile_matmat()" in docs
    assert "compile_rmatmat()" in docs
    assert "raw bound-method closure capture" in docs
    assert "outside the graph-memory guarantee" in docs
    assert "opaque and non-learnable" in docs
    assert "internal candidate" in docs
    assert "separate approved promotion plan" in docs
    assert "_PackedJaxLinearARG" not in docs
    assert "linear_dag.core.jaxlinarg.ingress" not in docs
    assert "HiJAX" not in docs


def test_jax_api_docs_match_backend_ingress_and_fallback_boundaries() -> None:
    docs = _normalized("docs/api/jax.md")

    assert "pure JAX" in docs
    assert "CPU FFI" in docs
    assert "no Pallas backend" in docs
    assert "GPU uses the portable pure-JAX path" in docs
    assert "does not define a packed serialization format" in docs
    assert "genoio" in docs
    assert "reconstruction, peak residency, transform, and schema-parity gates" in docs
    assert "Generic group fixtures are not durable Zarr integration coverage" in docs
    assert "max_padding_ratio=1.25" in docs
    assert "larger value or `None`" in docs
    assert "does not automatically fall back" in docs
    assert "JaxParallelOperator" in docs


def test_cli_and_readme_docs_preserve_exact_ragged_coexistence() -> None:
    cli_docs = _normalized("docs/cli.md")
    readme = _normalized("README.md")

    assert "`--jax-backend`" in cli_docs
    assert "RHE-only" in cli_docs
    assert "exact-ragged" in cli_docs
    assert "Backend.AUTO" in cli_docs
    assert "does not select the internal packed candidate" in cli_docs
    assert "docs/api/jax.md" in readme
    assert "public exact-ragged" in readme
    assert "internal packed candidate" in readme


def test_jax_api_page_is_in_site_navigation() -> None:
    navigation = _read_document("mkdocs.yml")

    assert "- JAX operators: api/jax.md" in navigation
