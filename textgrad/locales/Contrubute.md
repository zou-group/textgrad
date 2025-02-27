# Contribution Guide

## Adding a New Language

1. Create a new directory under `locales` with the language code (e.g., `locales/de/LC_MESSAGES/`).
2. Use xgettext to extract the strings from the source code and create a `.pot` file (e.g. `xgettext --language=Python --keyword=_ --output=locales/textgrad.pot textgrad/**/*.py`).
3. Copy the `.pot` file into this directory and rename it to `textgrad.po`.
4. Translate the strings in `textgrad.po`.
5. Compile the `.po` file into a `.mo` file using `msgfmt`.
6. Ensure the `textgrad/locales/[LANG]/textgrad.po` is included in the version control.

## Updating Existing Languages

1. Use xgettext to extract the strings just in the same way as that for adding new languages.
2. Merge the updated .pot file into existing .po files
    ```bash
    for lang in locales/*; do
        msgmerge --update $lang/LC_MESSAGES/textgrad.po locales/textgrad.pot
    done
    ```
3. Potentially outdated translations are marked as "fuzzy". Identify and revise them  accordingly.