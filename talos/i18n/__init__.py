"""
Internationalization (i18n) module for Talos
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

class I18n:
    """Internationalization handler"""
    
    def __init__(self, language: str = "en"):
        """
        Initialize i18n with specified language
        
        Args:
            language: Language code (en, ko, ja)
        """
        self.language = language
        self.translations: Dict[str, Any] = {}
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files"""
        i18n_dir = Path(__file__).parent
        translation_file = i18n_dir / f"{self.language}.json"
        
        if translation_file.exists():
            with open(translation_file, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        else:
            # Fallback to English if requested language not found
            fallback_file = i18n_dir / "en.json"
            if fallback_file.exists():
                with open(fallback_file, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
    
    def t(self, key: str, **kwargs) -> str:
        """
        Translate a key to current language
        
        Args:
            key: Translation key (dot notation supported, e.g., 'ui.task.create')
            **kwargs: Variables for string formatting
            
        Returns:
            Translated string
        """
        keys = key.split('.')
        value = self.translations
        
        try:
            for k in keys:
                value = value[k]
            
            # Handle string formatting
            if kwargs and isinstance(value, str):
                return value.format(**kwargs)
            
            return str(value)
        except (KeyError, TypeError):
            # Return key if translation not found
            return key
    
    def set_language(self, language: str):
        """
        Change current language
        
        Args:
            language: New language code
        """
        self.language = language
        self._load_translations()

# Global i18n instance
_i18n_instance = None

def get_i18n(language: str = None) -> I18n:
    """
    Get global i18n instance
    
    Args:
        language: Language code, if None uses existing instance language
        
    Returns:
        I18n instance
    """
    global _i18n_instance
    
    if _i18n_instance is None:
        _i18n_instance = I18n(language or "en")
    elif language and language != _i18n_instance.language:
        _i18n_instance.set_language(language)
    
    return _i18n_instance

def t(key: str, **kwargs) -> str:
    """
    Convenience function for translation
    
    Args:
        key: Translation key
        **kwargs: Variables for string formatting
        
    Returns:
        Translated string
    """
    return get_i18n().t(key, **kwargs)