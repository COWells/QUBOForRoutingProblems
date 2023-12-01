class InputGraphInvalidException(Exception):
    """
    Raised when the user input graph is invalid
    """
    pass


class MatrixNotSquareException(InputGraphInvalidException):
    """
    Raised when the user inputs a non-square matrix
    """
    pass


class MatrixNotSymmetricException(InputGraphInvalidException):
    """
    Raised when the user inputs a non-symmetric matrix
    """
    pass


class MatrixHasNonZeroLeadingDiagonal(InputGraphInvalidException):
    """
    Raised when the user has a non-zero element in the leading diagonal
    """
    pass


class MatrixHasZeroNotInLeadingDiagonal(InputGraphInvalidException):
    """
    Raised when the user has a zero element not in the leading diagonal
    """
    pass