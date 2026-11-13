"""The package can be imported: the least a build needs."""

import shader


def test_package_is_described() -> None:
    assert shader.__doc__
