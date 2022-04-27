"""
Miscellaneous utilities that do not fit nicely within other utils.

This is for testing only.
"""
import importlib
import inspect
import time
from functools import wraps
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from udeml.assets import b


def get_callable_kwargs(func: Callable) -> List[str]:
    """Get list of arguments for a callable.

    Args:
        func: A Callable object like a function or a class

    Returns:
        List of argument names accepted by the callable.
    """
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def filter_unsupported_kwargs(logger: Optional[Callable] = None):
    """Decorator for filtering out kwargs which are not supported.

    Args:
        logger: A callable that takes a string and logs it. In this case, it will
            only be used to warn about any kwargs that will be filtered out due to
            not being supported by the wrapped function.

    """

    def remove_wrong_kwargs_(func):
        func_kwargs = set(get_callable_kwargs(func))

        @wraps(func)
        def func_(*args, **kwargs):
            unsupported_kwargs = {
                k: v for k, v in kwargs.items() if k not in func_kwargs
            }
            if unsupported_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in func_kwargs}
                if logger:
                    logger(f"WARNING: filtering out {unsupported_kwargs}")
            return func(*args, **kwargs)

        return func_

    return remove_wrong_kwargs_


def reload_module(module_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return module


def retry(
    max_tries: int = 3,
    wait_secs: int = 5,
    raise_on_fail: bool = True,
    res_on_fail: Any = None,
) -> Any:
    """Decorator for allowing a function to retry

    Args:
        max_tries: Maximum number of attempts to call the function. (default: 1)
        wait_secs: time to wait between failures in seconds. (default: 5)
        raise_on_fail: Whether to raise if all attempts fail (default: True)
        res_on_fail: value to return in case of failure (default: None). Obviously,
            this only makes sense when raise_on_fail is set to False.
    """

    def retry_(func):
        @wraps(func)
        def retry_func(*args, **kwargs):
            tries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    print(f"ERROR: attempt={tries} failed in calling {func}: {e}")
                    if tries >= max_tries:
                        print(f"Max tries={max_tries} reached for {func}.")
                        if raise_on_fail:
                            raise e
                        else:
                            break
                    time.sleep(wait_secs)
                    print(f"Retrying to call {func}")
            return res_on_fail

        return retry_func

    return retry_


def diff_rec(
    left: Union[dict, Iterable],
    right: Union[dict, Iterable],
    ignore_keys: Optional[set] = None,
    comp_mode: str = "dp",
    precision: int = 6,
) -> Union[Dict, Tuple]:
    """Returns diff of two dictionaries (or non-string iterables).
    It uses recursive drill down, so do not use on structures which have big depth!

    Args:
        left: left dict or non-string iterable
        right: right dict or non-string iterable
        ignore_keys: optional set of keys to be ignored during comparison
        comp_mode: Numerical comparison modes. Must be one of
            - sf : significant figures
            - dp : decimal places
            - pc : percentage change
        precision: precision depending on the mode:
            if comp_mode='sf' then significant figures to round to before comparison
            if comp_mode='dp' then decimal places to round to before comparison
            if comp_mode='pc' then percentage change required to trigger diff

    Returns:
        diff of left and right. If left and right were dictionaries that are not
        identical, then one or more of the following keys will exist:
            - "ValDiff": if there are keys which have value differences, then this
                will contain a dictionary of those keys with values being tuple of
                (left_value, right_value, diff_value) if numerical or simply
                (left_value, right_value) if not numerical.
            - "OnlyInLeft": if there are some keys that only exist in the left, then
                those keys and their values will appear here.
            - "OnlyInLeft": if there are some keys that only exist in the right, then
                those keys and their values will appear here.
    """
    if ignore_keys is None:
        ignore_keys = set()
    ignore_keys = set(ignore_keys)

    def diff_rec_(left, right):
        diffs = {}
        left_is_dict = isinstance(left, dict)
        right_is_dict = isinstance(right, dict)
        if left_is_dict and right_is_dict:
            left_keys = set(left)
            right_keys = set(right)
            shared_keys = left_keys & right_keys - ignore_keys
            only_in_left_keys = left_keys - right_keys - ignore_keys
            only_in_right_keys = right_keys - left_keys - ignore_keys
            map_diffs = {}
            for key in shared_keys:
                res = diff_rec_(left[key], right[key])
                if res:
                    if "ValDiff" not in map_diffs:
                        map_diffs["ValDiff"] = {}
                    map_diffs["ValDiff"][key] = res
            if only_in_left_keys:
                tmp_map = {}
                for k in only_in_left_keys:
                    tmp_map[k] = left[k]
                map_diffs["OnlyInLeft"] = tmp_map
            if only_in_right_keys:
                tmp_map = {}
                for k in only_in_right_keys:
                    tmp_map[k] = right[k]
                map_diffs["OnlyInRight"] = tmp_map
            if map_diffs:
                diffs = map_diffs
        elif not left_is_dict and not right_is_dict:
            left_is_iterable = hasattr(left, "__iter__") and not isinstance(left, str)
            right_is_iterable = hasattr(right, "__iter__") and not isinstance(
                right, str
            )
            if left_is_iterable and right_is_iterable:
                iterable_diffs = {}
                idx = -1
                for item1, item2 in zip_longest(left, right):
                    idx += 1
                    res = diff_rec_(item1, item2)
                    if res:
                        iterable_diffs[idx] = res
                if iterable_diffs:
                    diffs = iterable_diffs
            elif not left_is_iterable and not right_is_iterable:
                if left != right:
                    if comp_mode == "sf":
                        try:
                            left_num = ("%%.%dg" % (int(precision))) % (left)
                            right_num = ("%%.%dg" % (int(precision))) % (right)
                            if left_num != right_num:
                                diffs = (left, right, right - left)
                        except:  # noqa: E722
                            diffs = (left, right)
                    elif comp_mode == "dp":
                        try:
                            left_num = ("{0:.%sf}" % (int(precision))).format(
                                float(left)
                            )
                            right_num = ("{0:.%sf}" % (int(precision))).format(
                                float(right)
                            )
                            if left_num != right_num:
                                diffs = (left, right, right - left)
                        except:  # noqa: E722
                            diffs = (left, right)
                    elif comp_mode == "pc":
                        try:
                            if abs(right - left) / float(abs(left)) >= precision:
                                diffs = (left, right, right - left)
                        except:  # noqa: E722
                            diffs = (left, right)
                    else:
                        raise Exception("comp_mode='%s' not supported" % (comp_mode))
            else:
                diffs = (left, right)
        else:
            diffs = (left, right)
        return diffs

    return diff_rec_(left, right)


def isfloat(x: str):
    """Determines whether given string is convertable to float or not.

    Args:
        x: input string to be checked

    Returns:
        True if the string is convertable to float, False otherwise.
    """
    try:
        float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x: str):
    """Determines whether given string is convertable to integer or not.

    Args:
        x: input string to be checked

    Returns:
        True if the string is convertable to int, False otherwise.
    """
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b


def convert_args_list_to_dict(args_list: List[str]) -> Dict:
    """Converts given arguments lists to a dictionary of argument name
    and value pairs.

    Args:
       args_list: A list containing of script arguments.
       It does not support boolean flags. All argument names should start with "--"
       and should be followed by the argument value.
       e.g. ['--test_value', 'A', '--num', '10']

    Returns:
        A dictionary of argument name and value pairs.
        e.g. {"test_value": "A", "num": 10}
    """
    assert len(args_list) % 2 == 0, (
        "The length of the list is not divisible by 2, "
        "make sure that all arguments have name-value pairs."
    )

    args_dict = {}
    for i in range(0, len(args_list), 2):
        argument_name = args_list[i]
        argument_value = args_list[i + 1]

        assert argument_name.startswith(
            "--"
        ), "Argument name must start will double dashes."
        assert not argument_value.startswith(
            "--"
        ), "Argument value must not start with double dashes."

        if isint(argument_value):
            argument_value = int(argument_value)
        elif isfloat(argument_value):
            argument_value = float(argument_value)

        argument_name = argument_name[2:]  # Remove the dashes from the name
        args_dict[argument_name] = argument_value

    return args_dict


def add_dashes_to_arg_names(args_list: List[str]) -> List[str]:
    """Adds dashes to the argument names in a list of argument names and values

    Args:
       args_list: A list containing of script arguments.
       Argument name should not start with "--"
       and should be followed by the argument value.
       e.g. ['test_value', 'A', 'num', '10']

    Returns:
        A list of argument names with dashes and corresponding values.
        e.g. ['--test_value', 'A', '--num', '10']
    """
    assert len(args_list) % 2 == 0, (
        "The length of the list is not divisible by 2, "
        "make sure that all arguments have name-value pairs."
    )

    for i in range(0, len(args_list), 2):
        assert not args_list[i].startswith(
            "--"
        ), "Argument name must not start will double dashes."
        args_list[i] = "--" + args_list[i]

    return args_list


def remove_dashes_from_arg_names_in_dict(args_dict: Dict) -> Dict:
    """Removes dashes from the argument names in a dictionary of argument names and values

    Args:
       args_dict: A dictionary containing of script arguments.
       Argument name should start with "--"
       and should be followed by the argument value.
       e.g. {'--test_value':'A', '--num':'10'}

    Returns:
        A dictionary of argument names without dashes and corresponding values.
        e.g. {'test_value':'A', '--num':'10'}
    """
    new_args_dict = {}
    for argument_name, argument_value in args_dict.items():
        assert argument_name.startswith(
            "--"
        ), "Argument name must start will double dashes."
        argument_name = argument_name[2:]  # Remove the dashes from the name
        new_args_dict[argument_name] = argument_value

    return new_args_dict
