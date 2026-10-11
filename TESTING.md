# Testing

The repository includes lightweight tests that do not contact Sentinel Hub and do not require credentials.

Run them locally with:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall scripts tests
python -m pytest -q
```

The tests check that the public scripts parse as Python, that Sentinel Hub credentials are read from environment variables, that the article-aligned parameters are present, and that the RF/SVM benchmark script keeps the manual benchmark separate from the pseudo-label transfer check.
