class SkopsIoException(Exception):
    """Base class for skops.io errors"""


class UnsupportedTypeException(TypeError, SkopsIoException):
    """Raise when an object of this type is known to be unsupported"""

    def __init__(self, obj):
        super().__init__(
            f"Objects of type {obj.__class__.__name__} are not supported yet."
        )


class UnsafeTypeError(TypeError, SkopsIoException):
    """Raise when it's known that this type might be unsafe"""

    def __init__(self, cls_name):
        super().__init__(f"Objects of type {cls_name} are potentially unsafe")
