# Orakl

---

Orakl is a active learning library for TensorFlow.

## Installation
**Installation Requirements**
* python >= 3.6
* tensorflow>=2.0.0

#### Manual

If you'd like to try our latest features, you can install the latest master directly from GitHub. For a basic install, run:

```bash
git clone https://github.com/nocotan/orakl.git
cd orakl
pip install -e .
```

## Tests

To execute unit tests from a manual install, run:

```bash
python3 -m unittest tests/attr/core/strategies/test_expected_model_change.py
```

or recursive test execution via the command line is built-in:

```bash
python3 -m unittest discover tests
```

## LICENSE
Orakl is Apache-2.0 licensed, as found in the [LICENSE](LICENSE) file.