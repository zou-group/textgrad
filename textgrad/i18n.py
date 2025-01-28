"""
This module provides functions for language localization in the TextGrad package.
"""
import logging
import gettext
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _set_language(lang=None):
    """
    Set the language for localization.

    Note:
    - This function does not support changing the language dynamically after it has been set.
    - The language is determined when the module is first imported by checking the following
      environment variables: LANGUAGE, LC_ALL, LC_MESSAGES, and LANG.

    Args:
        lang (str, optional): The language code to set. Defaults to None.
    """
    locale_dir = os.path.join(os.path.dirname(__file__), 'locales')
    gettext.bindtextdomain('textgrad', locale_dir)
    gettext.textdomain('textgrad')
    if lang is not None:
        lang = [lang]
    try:
        translation = gettext.translation('textgrad', locale_dir, languages=lang)
    except FileNotFoundError:
        if lang and lang != 'en':
            logger.warning("Language '%s' not found under %s. Using fallback options.",
                           lang, locale_dir)
        translation = gettext.translation('textgrad', locale_dir, languages=lang, fallback=True)
    translation.install()
