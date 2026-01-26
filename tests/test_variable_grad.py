import pytest
from textgrad.variable import Variable

def test_requires_grad_validation():
    # Case 1: Predecessor has requires_grad=True
    v1 = Variable("input", requires_grad=True, role_description="input")
    
    # Case 2: Successor initially has requires_grad=True (default)
    v2 = Variable("output", predecessors=[v1], role_description="output")
    assert v2.requires_grad is True
    
    # Case 3: Try to set successor requires_grad=False -> Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        v2.requires_grad = False
    
    assert "If the variable does not require grad" in str(excinfo.value)

def test_requires_grad_validation_init():
    # Test initialization validation (if logic was added to __init__ as well, 
    # but my refactor only added property setter validation logic.
    # The __init__ calls self._requires_grad directly in my implementation 
    # or relies on the logic. Wait, let's check my implementation.)
    # In my implementation, I set self._requires_grad = requires_grad in __init__.
    # So __init__ bypasses the setter check. 
    # But wait, the original code had a check in __init__:
    # if (not requires_grad) and (len(_predecessor_requires_grad) > 0): raise Exception...
    # I should verify that this check is still preserved or if I need to rely on it.
    
    v1 = Variable("input", requires_grad=True, role_description="input")
    
    # Init with requires_grad=False should fail
    with pytest.raises(Exception) as excinfo: # Original code raised Exception, not ValueError
        v2 = Variable("output", predecessors=[v1], role_description="output", requires_grad=False)
    
    assert "If the variable does not require grad" in str(excinfo.value)
