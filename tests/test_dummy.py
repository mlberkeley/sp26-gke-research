from sp26_gke import hello


def test_hello() -> None:
    assert hello() == "sp26-google-gke-research"
