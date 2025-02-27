import builtins


def return_as_is(text):
    """This function is to avoid the undefined _ from ruining tests"""
    return text


builtins.__dict__['_'] = return_as_is
