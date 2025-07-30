"""
Utility functions for sci_demo framework.
"""

import weave
from typing import Any, Dict

def is_weave_initialized() -> bool:
    """Check if weave has been initialized by the user."""
    try:
        import weave
        # Use the official get_client() function to check if weave is initialized
        client = weave.get_client()
        return client is not None
    except Exception:
        return False

def safe_weave_log(data: Dict[str, Any]) -> None:
    """Safely log to weave call summary only if it's been initialized."""
    if not is_weave_initialized():
        return
    
    try:
        # Get the current call context using weave's internal API
        call = weave.get_current_call()
        if call is not None:
            # Update the call summary with the provided data
            if hasattr(call, 'summary') and call.summary is not None:
                call.summary.update(data)
    except Exception:
        # Silently ignore if weave is not properly initialized or call context unavailable
        pass
