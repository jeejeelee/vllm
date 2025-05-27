# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
import json
import sys
import textwrap
import warnings
from argparse import (Action, ArgumentDefaultsHelpFormatter, ArgumentParser,
                      ArgumentTypeError, Namespace,
                      RawDescriptionHelpFormatter, _ArgumentGroup)
from collections import defaultdict
from collections.abc import Mapping
from functools import partial, wraps
from typing import Any, Callable, Optional, TypeVar, Union

import regex as re
import yaml

from vllm.logger import init_logger

logger = init_logger(__name__)


class StoreBoolean(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() == "true":
            setattr(namespace, self.dest, True)
        elif values.lower() == "false":
            setattr(namespace, self.dest, False)
        else:
            raise ValueError(f"Invalid boolean value: {values}. "
                             "Expected 'true' or 'false'.")


class SortedHelpFormatter(ArgumentDefaultsHelpFormatter,
                          RawDescriptionHelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def _split_lines(self, text, width):
        """
        1. Sentences split across lines have their single newlines removed.
        2. Paragraphs and explicit newlines are split into separate lines.
        3. Each line is wrapped to the specified width (width of terminal).
        """
        # The patterns also include whitespace after the newline
        single_newline = re.compile(r"(?<!\n)\n(?!\n)\s*")
        multiple_newlines = re.compile(r"\n{2,}\s*")
        text = single_newline.sub(' ', text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


class FlexibleArgumentParser(ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    _deprecated: set[Action] = set()

    def __init__(self, *args, **kwargs):
        # Set the default 'formatter_class' to SortedHelpFormatter
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = SortedHelpFormatter
        super().__init__(*args, **kwargs)

    if sys.version_info < (3, 13):
        # Enable the deprecated kwarg for Python 3.12 and below

        def parse_known_args(self, args=None, namespace=None):
            namespace, args = super().parse_known_args(args, namespace)
            for action in FlexibleArgumentParser._deprecated:
                if (hasattr(namespace, dest := action.dest)
                        and getattr(namespace, dest) != action.default):
                    logger.warning_once("argument '%s' is deprecated", dest)
            return namespace, args

        def add_argument(self, *args, **kwargs):
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                FlexibleArgumentParser._deprecated.add(action)
            return action

        class _FlexibleArgumentGroup(_ArgumentGroup):

            def add_argument(self, *args, **kwargs):
                deprecated = kwargs.pop("deprecated", False)
                action = super().add_argument(*args, **kwargs)
                if deprecated:
                    FlexibleArgumentParser._deprecated.add(action)
                return action

        def add_argument_group(self, *args, **kwargs):
            group = self._FlexibleArgumentGroup(self, *args, **kwargs)
            self._action_groups.append(group)
            return group

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: Namespace | None = None,
    ):
        if args is None:
            args = sys.argv[1:]

        # Check for --model in command line arguments first
        if args and args[0] == "serve":
            model_in_cli_args = any(arg == '--model' for arg in args)

            if model_in_cli_args:
                raise ValueError(
                    "With `vllm serve`, you should provide the model as a "
                    "positional argument or in a config file instead of via "
                    "the `--model` option.")

        if '--config' in args:
            args = self._pull_args_from_config(args)

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = '--' + key[len('--'):].replace('_', '-')
                    processed_args.append(f'{key}={value}')
                else:
                    processed_args.append('--' +
                                          arg[len('--'):].replace('_', '-'))
            elif arg.startswith('-O') and arg != '-O' and len(arg) == 2:
                # allow -O flag to be used without space, e.g. -O3
                processed_args.append('-O')
                processed_args.append(arg[2:])
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: list[str], value: str):
            """Creates a nested dictionary from a list of keys and a value.

            For example, `keys = ["a", "b", "c"]` and `value = 1` will create:
            `{"a": {"b": {"c": 1}}}`
            """
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(original: dict, update: dict):
            """Recursively updates a dictionary with another dictionary."""
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    recursive_dict_update(original[k], v)
                else:
                    original[k] = v

        delete = set()
        dict_args: dict[str, dict] = defaultdict(dict)
        for i, processed_arg in enumerate(processed_args):
            if processed_arg.startswith("--") and "." in processed_arg:
                if "=" in processed_arg:
                    processed_arg, value = processed_arg.split("=", 1)
                    if "." not in processed_arg:
                        # False positive, . was only in the value
                        continue
                else:
                    value = processed_args[i + 1]
                    delete.add(i + 1)
                key, *keys = processed_arg.split(".")
                # Merge all values with the same key into a single dict
                arg_dict = create_nested_dict(keys, value)
                recursive_dict_update(dict_args[key], arg_dict)
                delete.add(i)
        # Filter out the dict args we set to None
        processed_args = [
            a for i, a in enumerate(processed_args) if i not in delete
        ]
        # Add the dict args back as if they were originally passed as JSON
        for dict_arg, dict_value in dict_args.items():
            processed_args.append(dict_arg)
            processed_args.append(json.dumps(dict_value))

        return super().parse_args(processed_args, namespace)

    def check_port(self, value):
        try:
            value = int(value)
        except ValueError:
            msg = "Port must be an integer"
            raise ArgumentTypeError(msg) from None

        if not (1024 <= value <= 65535):
            raise ArgumentTypeError("Port must be between 1024 and 65535")

        return value

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tensor-parallel-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        assert args.count(
            '--config') <= 1, "More than one config file specified!"

        index = args.index('--config')
        if index == len(args) - 1:
            raise ValueError("No config file specified! \
                             Please check your command-line arguments.")

        file_path = args[index + 1]

        config_args = self._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # optionally followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith('-')
            model_in_config = any(arg == '--model' for arg in config_args)

            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file.")

            if model_in_cli:
                # Model specified as positional arg, keep CLI version
                args = [args[0]] + [
                    args[1]
                ] + config_args + args[2:index] + args[index + 2:]
            else:
                # No model in CLI, use config if available
                args = [args[0]
                        ] + config_args + args[1:index] + args[index + 2:]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2:]

        return args

    def _load_config_file(self, file_path: str) -> list[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tensor-parallel-size': '4'
            ]
        """
        extension: str = file_path.split('.')[-1]
        if extension not in ('yaml', 'yml'):
            raise ValueError(
                "Config file must be of a yaml/yml type.\
                              %s supplied", extension)

        # only expecting a flat dictionary of atomic types
        processed_args: list[str] = []

        config: dict[str, Union[int, str]] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. \
                Make sure path is correct", file_path)
            raise ex

        store_boolean_arguments = [
            action.dest for action in self._actions
            if isinstance(action, StoreBoolean)
        ]

        for key, value in config.items():
            if isinstance(value, bool) and key not in store_boolean_arguments:
                if value:
                    processed_args.append('--' + key)
            else:
                processed_args.append('--' + key)
                processed_args.append(str(value))

        return processed_args


T = TypeVar("T")


# `functools` helpers
def identity(value: T, **kwargs) -> T:
    """Returns the first provided value."""
    return value


F = TypeVar('F', bound=Callable[..., Any])


def deprecate_args(
    start_index: int,
    is_deprecated: Union[bool, Callable[[], bool]] = True,
    additional_message: Optional[str] = None,
) -> Callable[[F], F]:

    if not callable(is_deprecated):
        is_deprecated = partial(identity, is_deprecated)

    def wrapper(fn: F) -> F:

        params = inspect.signature(fn).parameters
        pos_types = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        pos_kws = [
            kw for kw, param in params.items() if param.kind in pos_types
        ]

        @wraps(fn)
        def inner(*args, **kwargs):
            if is_deprecated():
                deprecated_args = pos_kws[start_index:len(args)]
                if deprecated_args:
                    msg = (
                        f"The positional arguments {deprecated_args} are "
                        "deprecated and will be removed in a future update.")
                    if additional_message is not None:
                        msg += f" {additional_message}"

                    warnings.warn(
                        DeprecationWarning(msg),
                        stacklevel=3,  # The inner function takes up one level
                    )

            return fn(*args, **kwargs)

        return inner  # type: ignore

    return wrapper


def supports_kw(
    callable: Callable[..., object],
    kw_name: str,
    *,
    requires_kw_only: bool = False,
    allow_var_kwargs: bool = True,
) -> bool:
    """Check if a keyword is a valid kwarg for a callable; if requires_kw_only
    disallows kwargs names that can also be positional arguments.
    """
    params = inspect.signature(callable).parameters
    if not params:
        return False

    param_val = params.get(kw_name)

    # Types where the it may be valid, i.e., explicitly defined & nonvariadic
    passable_kw_types = set((inspect.Parameter.POSITIONAL_ONLY,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY))

    if param_val:
        is_sig_param = param_val.kind in passable_kw_types
        # We want kwargs only, but this is passable as a positional arg
        if (requires_kw_only and is_sig_param
                and param_val.kind != inspect.Parameter.KEYWORD_ONLY):
            return False
        if ((requires_kw_only
             and param_val.kind == inspect.Parameter.KEYWORD_ONLY)
                or (not requires_kw_only and is_sig_param)):
            return True

    # If we're okay with var-kwargs, it's supported as long as
    # the kw_name isn't something like *args, **kwargs
    if allow_var_kwargs:
        # Get the last param; type is ignored here because params is a proxy
        # mapping, but it wraps an ordered dict, and they appear in order.
        # Ref: https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
        last_param = params[next(reversed(params))]  # type: ignore
        return (last_param.kind == inspect.Parameter.VAR_KEYWORD
                and last_param.name != kw_name)
    return False


def deprecate_kwargs(
    *kws: str,
    is_deprecated: Union[bool, Callable[[], bool]] = True,
    additional_message: Optional[str] = None,
) -> Callable[[F], F]:
    deprecated_kws = set(kws)

    if not callable(is_deprecated):
        is_deprecated = partial(identity, is_deprecated)

    def wrapper(fn: F) -> F:

        @wraps(fn)
        def inner(*args, **kwargs):
            if is_deprecated():
                deprecated_kwargs = kwargs.keys() & deprecated_kws
                if deprecated_kwargs:
                    msg = (
                        f"The keyword arguments {deprecated_kwargs} are "
                        "deprecated and will be removed in a future update.")
                    if additional_message is not None:
                        msg += f" {additional_message}"

                    warnings.warn(
                        DeprecationWarning(msg),
                        stacklevel=3,  # The inner function takes up one level
                    )

            return fn(*args, **kwargs)

        return inner  # type: ignore

    return wrapper


def resolve_mm_processor_kwargs(
    init_kwargs: Optional[Mapping[str, object]],
    inference_kwargs: Optional[Mapping[str, object]],
    callable: Callable[..., object],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """Applies filtering to eliminate invalid mm_processor_kwargs, i.e.,
    those who are not explicit keywords to the given callable (of one is
    given; otherwise no filtering is done), then merges the kwarg dicts,
    giving priority to inference_kwargs if there are any collisions.

    In the case that no kwarg overrides are provided, returns an empty
    dict so that it can still be kwarg expanded into the callable later on.

    If allow_var_kwargs=True, allows for things that can be expanded into
    kwargs as long as they aren't naming collision for var_kwargs or potential
    positional arguments.
    """
    # Filter inference time multimodal processor kwargs provided
    runtime_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=inference_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Filter init time multimodal processor kwargs provided
    init_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=init_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Merge the final processor kwargs, prioritizing inference
    # time values over the initialization time values.
    mm_processor_kwargs = {**init_mm_kwargs, **runtime_mm_kwargs}
    return mm_processor_kwargs


def get_allowed_kwarg_only_overrides(
    callable: Callable[..., object],
    overrides: Optional[Mapping[str, object]],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """
    Given a callable which has one or more keyword only params and a dict
    mapping param names to values, drop values that can be not be kwarg
    expanded to overwrite one or more keyword-only args. This is used in a
    few places to handle custom processor overrides for multimodal models,
    e.g., for profiling when processor options provided by the user
    may affect the number of mm tokens per instance.

    Args:
        callable: Callable which takes 0 or more keyword only arguments.
                  If None is provided, all overrides names are allowed.
        overrides: Potential overrides to be used when invoking the callable.
        allow_var_kwargs: Allows overrides that are expandable for var kwargs.

    Returns:
        Dictionary containing the kwargs to be leveraged which may be used
        to overwrite one or more keyword only arguments when invoking the
        callable.
    """
    if not overrides:
        return {}

    # Drop any mm_processor_kwargs provided by the user that
    # are not kwargs, unless it can fit it var_kwargs param
    filtered_overrides = {
        kwarg_name: val
        for kwarg_name, val in overrides.items()
        if supports_kw(callable,
                       kwarg_name,
                       requires_kw_only=requires_kw_only,
                       allow_var_kwargs=allow_var_kwargs)
    }

    # If anything is dropped, log a warning
    dropped_keys = overrides.keys() - filtered_overrides.keys()
    if dropped_keys:
        if requires_kw_only:
            logger.warning(
                "The following intended overrides are not keyword-only args "
                "and will be dropped: %s", dropped_keys)
        else:
            logger.warning(
                "The following intended overrides are not keyword args "
                "and will be dropped: %s", dropped_keys)

    return filtered_overrides
