# 🏗️ Framework Design: Proper Weave Integration

## Overview

`sci_demo` is designed as a **framework** for building iterative AI science workflows, not as a standalone application. This design choice has important implications for how Weave tracking is integrated.

## Key Design Principles

### 1. **User Controls Project Naming** 🎯
- **Framework responsibility**: Provide tracking capabilities
- **User responsibility**: Initialize weave with their project name
- **Why**: Different users/organizations need different project names

```python
# ✅ CORRECT - User controls the project
import weave
weave.init("my_research_project_name")  # User choice

from sci_demo import Generation, Workflow
```

```python
# ❌ WRONG - Framework forces project name
from sci_demo import Generation  # This would auto-init "sci_demo_project"
```

### 2. **Optional Tracking** ⚪
- Framework works with or without Weave tracking
- Users can disable tracking by simply not calling `weave.init()`
- All `@weave.op()` decorators remain in place but become no-ops

```python
# Tracking enabled
import weave
weave.init("my_project")
from sci_demo import Workflow  # All operations tracked

# Tracking disabled  
from sci_demo import Workflow  # No tracking, everything still works
```

### 3. **Framework vs Application** 🏗️

| Framework (sci_demo) | Application |
|---|---|
| Users control initialization | App controls initialization |
| Multiple project names possible | Single project name |
| Optional tracking | Built-in tracking |
| Imported by users | Run directly |

## Implementation Details

### Module-Level Design

**Before (❌ Inappropriate for framework):**
```python
# sci_demo/workflow.py
import weave
weave.init("workflow")  # Forces project name!

class Workflow:
    # ...
```

**After (✅ Framework-appropriate):**
```python
# sci_demo/workflow.py  
import weave
# No weave.init() at module level

class Workflow:
    @weave.op()  # Decorator stays - becomes no-op if weave not initialized
    def run(self):
        # ...
```

### Testing Each Module

Each module can still be tested individually:

```python
# sci_demo/workflow.py
if __name__ == "__main__":
    # Only for testing this module
    weave.init("workflow_module_test")
    
    # Test code here...
```

### Safe Weave Utilities

The framework provides utilities for graceful degradation:

```python
# sci_demo/__init__.py
def is_weave_initialized() -> bool:
    """Check if user has initialized weave."""
    
def safe_weave_log(data: Dict) -> None:
    """Log only if weave is initialized."""
```

## User Workflow

### Typical Usage Pattern

```python
# Step 1: User sets up their environment
import weave
weave.init("quantum_chemistry_research_2024")

# Step 2: Import framework components  
from sci_demo import Generation, Workflow, Oracle, Prompt

# Step 3: Build their application
workflow = Workflow(...)
result = workflow.run(...)

# Step 4: Check their dashboard
# All tracking appears under "quantum_chemistry_research_2024"
```

### Advanced Usage

```python
# A/B testing different configurations
for config in configs:
    weave.init(f"experiment_{config['name']}")
    # Each config gets its own project
    
# Multi-user organization
weave.init(f"{user_name}_research_project")  
# Each user gets their own project
```

## Benefits of This Design

### ✅ **Flexibility**
- Users control project organization
- Works in any environment (research, production, testing)
- Optional tracking reduces dependencies

### ✅ **Proper Framework Behavior**
- No side effects when importing
- User controls when/how tracking happens
- Multiple instances possible

### ✅ **Scalability** 
- Enterprise organizations can have their own naming schemes
- Different research groups can track separately
- A/B testing and experiment comparison easy

### ✅ **Backward Compatibility**
- Existing code keeps working
- All tracking features still available
- No breaking changes to API

## Migration Guide

If you have existing code that relied on automatic weave initialization:

### Before:
```python
from sci_demo import Workflow  # Auto-initialized weave
```

### After:
```python
import weave
weave.init("my_project_name")  # YOU control the name
from sci_demo import Workflow  # Now tracked under your project
```

## Framework Best Practices

### For Framework Developers
1. **Never call `weave.init()` at module level** in framework code
2. **Keep all `@weave.op()` decorators** - they become no-ops if not initialized
3. **Use `safe_weave_log()`** for optional logging
4. **Document user responsibilities** clearly

### For Framework Users
1. **Call `weave.init()` first** with your project name
2. **Import framework components after** weave initialization
3. **Check documentation** for weave-enabled features
4. **Use tracking strategically** - enable in research, optionally disable in production

## Examples

- **Research Lab**: `weave.init("stanford_ai_lab_protein_folding")`
- **Company**: `weave.init("pharma_corp_drug_discovery_2024")`  
- **Personal**: `weave.init("my_quantum_computing_experiments")`
- **A/B Testing**: `weave.init(f"experiment_{experiment_id}")`

This design ensures that `sci_demo` behaves as a proper framework while still providing comprehensive tracking capabilities when desired. 