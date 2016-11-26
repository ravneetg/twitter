"""
Custom exceptions and warnings used for textauger.
"""

class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling. Adopted from scikit-learn, which uses the same exception.
    """
