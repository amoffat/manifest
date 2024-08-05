def pytest_addoption(parser):
    parser.addoption(
        "--check-garbage",
        action="store_true",
        default=False,
        help="Check remote service terms for various 'closed output' clauses",
    )
