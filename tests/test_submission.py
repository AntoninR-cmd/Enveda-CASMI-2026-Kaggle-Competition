"""Tests unitaires de la soumission."""

from casmi.submission import _format_smiles


def test_format_smiles():
    result = _format_smiles(["CCO", "CCC", "CCO"])
    assert result == "CCO;CCC"


def test_format_smiles_max_rank():
    result = _format_smiles(["S" + str(i) for i in range(30)])
    assert len(result.split(";")) == 25


def test_format_smiles_empty():
    assert _format_smiles([]) == ""
