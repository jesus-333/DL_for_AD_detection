# Coding Guidelines

Here are some guidelines I want you to follow when writing code.

## Dividers

When you need to separate code blocks, use the following string:
```python
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

E.g. use it to separate imports from function declarations.

When you are inside a function, use the following string instead (respecting indentation accordingly):
```python
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

Always try to divide each long file into thematic sections (e.g. imports, main functions, helper functions, etc.).  
The same applies to long functions — split them into thematic blocks (e.g. input checks, preprocessing, actual computations, output checks).  
Very long functions are better split into specific sub-functions.

## Style Notes

- Always use snake_case (including for classes).
- Variable names must be descriptive (even if long — clarity about their purpose is what matters).
- Always put spaces around equals signs, both for operations between variables and when passing arguments to functions.
    - E.g. (1): `a=1+2` → `a = 1 + 2`
    - E.g. (2): `function(a=2)` → `function(a = 2)`
- When specifying the type of an input variable or the return type of a function, always put a space before the colon.
    - E.g.: `def funct(a: int) -> int:` → `def funct(a : int) -> int :`
- The same applies to `if` statements — leave a space between the condition and `:`.
    - E.g.: `if condition:` → `if condition :`
- Comment the code inside functions as much as possible.
- Align equals signs where possible.
    - E.g. when declaring a dictionary with many entries, align the equals signs for each key.
    - E.g. when calling a function split across multiple lines with one argument per line, align the equals signs.
- When using `argparse`, provide an exhaustive description for each argument, the input type, and a default value if the argument is optional.
- When you add new `cli`, if they are made up of multiple word, use hyphen to separate them.
    - E.g. `my-multi-word-cli-command`
- Use always f-strig, for string with variable input (i.e. in the format `f"The value of x is {x}"`)

## Docstrings and comments

- Always use the **numpydoc** format for docstrings.
- In the future, `sphinx` will be used to auto-generate documentation. When referencing other modules/classes/functions inside docstrings, use the sphinx syntax that enables hyperlinks. For long links, use `~` to abbreviate the hyperlink to just its last component.
- Even for single-line docstrings, always add a newline after the opening `"""` (see example below).
- If you write a long comments/docstrings start a new line only after a period. Do not start a new line in the middle of a sentence.
- If you have short sentences you can put it in the same line.
- If you write word that can be written with both `s` and `z` use the version with `z` (e.g. optimized instead of optimized)

## Short Docstring Example

```python
""" This is a WRONG short docstring """
```

```python
"""
This is a RIGHT short docstring
"""
```


# Example

Here you will find a concrete example.

This is some python code that you wrote
```python
"""Command line entry point for Jesus's LLM Tools (JLT).

JLT is a collection of LLM-powered tools. Every tool is invoked through the
single ``jlt`` command using the syntax::

    jlt <tool_name> <tool_subcommand, tool variable, tool flags>

Each tool lives in its own subpackage under ``jlt`` and exposes a
``register(subparsers)`` function in its ``cli`` module. That function attaches
the tool's own argument parser (with its subcommands, variables and flags) to
the shared top-level parser, keeping every tool self-contained.
"""

from __future__ import annotations

import argparse
from importlib import import_module

from . import __version__

# Each entry maps a tool name to the module that exposes ``register(subparsers)``.
# Adding a new tool is a one-line change here.
TOOLS = (
    "jlt.autoresearch.cli",
    "jlt.club.cli",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``jlt`` parser and let every tool register itself."""
    parser = argparse.ArgumentParser(
        prog="jlt",
        description="Jesus's LLM Tools (JLT): a collection of LLM-powered tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="tool",
        metavar="<tool_name>",
        help="The JLT tool to run.",
    )
    subparsers.required = True

    for module_path in TOOLS:
        module = import_module(module_path)
        module.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected tool.

    Returns the tool's exit code (defaults to ``0``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # ``func`` is set by each tool via ``set_defaults`` in its ``register``.
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

This is the same code with my style rules.
```python
"""
Command line entry point for Jesus's LLM Tools (JLT).

JLT is a collection of LLM-powered tools. Every tool is invoked through the single ``jlt`` command using the syntax::

    jlt <tool_name> <tool_subcommand, tool variable, tool flags>

Each tool lives in its own subpackage under ``jlt`` and exposes a ``register(subparsers)`` function in its ``cli`` module. 
That function attaches the tool's own argument parser (with its subcommands, variables and flags) to the shared top-level parser, keeping every tool self-contained.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports sectio

from __future__ import annotations

# Full module imports
import argparse

# Specific imports
from importlib import import_module

# Internal imports
from . import __version__

# Each entry maps a tool name to the module that exposes ``register(subparsers)``.
# Adding a new tool is a one-line change here.
TOOLS = (
    "jlt.autoresearch.cli",
    "jlt.club.cli",
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Support function

def build_parser() -> argparse.ArgumentParser :
    """
    Build the top-level ``jlt`` parser and let every tool register itself.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Build jlt parser

    parser = argparse.ArgumentParser(
        prog        = "jlt",
        description = "Jesus's LLM Tools (JLT): a collection of LLM-powered tools.",
    )

    parser.add_argument(
        "--version",
        action  = "version",
        version = f"%(prog)s {__version__}",
    )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Build tool's parser

    subparsers = parser.add_subparsers(
        dest    = "tool",
        metavar = "<tool_name>",
        help    = "The JLT tool to run.",
    )
    subparsers.required = True

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for module_path in TOOLS:
        module = import_module(module_path)
        module.register(subparsers)

    return parser

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main function

def main(argv : list[str] | None = None) -> int :
    """
    Parse ``argv`` and dispatch to the selected tool.

    Returns the tool's exit code (defaults to ``0``).
    """
    
    # Get input argument
    parser  = build_parser()
    args    = parser.parse_args(argv)

    # Note that ``func`` is set by each tool via ``set_defaults`` in its ``register``.
    return args.func(args) or 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

```
