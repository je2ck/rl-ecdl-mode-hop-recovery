# Contributing

Contributions are welcome through focused pull requests.

1. Create a branch from `main`.
2. Keep laboratory addresses, credentials, empirical data, checkpoints, and
   generated plots out of commits.
3. Run `python scripts/audit_public_tree.py`,
   `python -m unittest discover -s tests -v`, and `cargo test --locked` from
   `simulator/` before opening a pull request.
4. Describe any change to the state representation, reward, action protocol, or
   simulator transition model explicitly. These changes can alter the scientific
   interpretation of a trained policy even when the API remains unchanged.

Hardware changes should be tested first with dummy clients or disconnected unit
tests. A pull request must never include a real controller address or VISA
resource identifier.
