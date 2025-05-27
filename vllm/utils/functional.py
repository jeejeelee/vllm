from typing import Callable,Any,Union
import weakref
from typing_extensions import ParamSpec
import inspect
from vllm.logger import init_logger
from functools import wraps
import cloudpickle
from functools import partial

logger = init_logger(__name__)


P = ParamSpec('P')

def weak_bind(bound_method: Callable[..., Any], ) -> Callable[..., None]:
    """Make an instance method that weakly references
    its associated instance and no-ops once that
    instance is collected."""
    ref = weakref.ref(bound_method.__self__)  # type: ignore[attr-defined]
    unbound = bound_method.__func__  # type: ignore[attr-defined]

    def weak_bound(*args, **kwargs) -> None:
        if inst := ref():
            unbound(inst, *args, **kwargs)

    return weak_bound


#From: https://stackoverflow.com/a/4104188/2749989
def run_once(f: Callable[P, None]) -> Callable[P, None]:

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if not wrapper.has_run:  # type: ignore[attr-defined]
            wrapper.has_run = True  # type: ignore[attr-defined]
            return f(*args, **kwargs)

    wrapper.has_run = False  # type: ignore[attr-defined]
    return wrapper

def warn_for_unimplemented_methods(cls: type[T]) -> type[T]:
    """
    A replacement for `abc.ABC`.
    When we use `abc.ABC`, subclasses will fail to instantiate
    if they do not implement all abstract methods.
    Here, we only require `raise NotImplementedError` in the
    base class, and log a warning if the method is not implemented
    in the subclass.
    """

    original_init = cls.__init__

    def find_unimplemented_methods(self: object):
        unimplemented_methods = []
        for attr_name in dir(self):
            # bypass inner method
            if attr_name.startswith('_'):
                continue

            try:
                attr = getattr(self, attr_name)
                # get the func of callable method
                if callable(attr):
                    attr_func = attr.__func__
            except AttributeError:
                continue
            src = inspect.getsource(attr_func)
            if "NotImplementedError" in src:
                unimplemented_methods.append(attr_name)
        if unimplemented_methods:
            method_names = ','.join(unimplemented_methods)
            msg = (f"Methods {method_names} not implemented in {self}")
            logger.warning(msg)

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        find_unimplemented_methods(self)

    type.__setattr__(cls, '__init__', wrapped_init)
    return cls

def run_method(obj: Any, method: Union[str, bytes, Callable], args: tuple[Any],
               kwargs: dict[str, Any]) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(f"Method {method!r} is not"
                                      " implemented.") from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)