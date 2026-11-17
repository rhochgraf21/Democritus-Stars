"""The package can be imported: the least a build needs."""

import democritus


def test_package_is_described() -> None:
    assert democritus.__doc__
