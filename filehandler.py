#-------------------------------------------------------------------------------
# Name:        file class with the additional option of autonatically open
# Purpose:     for purposes where a file must be modified after closed (i.e.)
#              after a context manager, without having to care how to
#              explicitely open the same file again.
# Author:      atiefenauer
#
# Created:     13.05.2014
# Copyright:   (c) atiefenauer 2014
#-------------------------------------------------------------------------------
#!/usr/bin/env python

__author__ = 'atiefenauer'

# ------------------------------------------------------------------------------
# DESCRIPTION:
#
# The file wrapper _FileWrapper makes sure a closed file is open automatically
# for the time of I/O and closed right afterwards again. This way the file is
# not opened between the writtings. Do not use this class for readings in
# combination with 'tell' would not work properly with the option forceopen=True
# Since the file is re-opended for individual I/O actions, 'tell' would always
# result to 0.
# ------------------------------------------------------------------------------
from functools import wraps
from tempfile import _TemporaryFileWrapper


def forceopen_decorator(func):
    @wraps(func)
    def forceopen_(fileobj, *args, **kwargs):
        if fileobj.closed and fileobj.forceopen:
            fname = getname(fileobj)
            fid = openforced(fname, mode=fileobj.mode, forceopen=True)
            out = func(fid, *args, **kwargs)
            return out
        else:
            return func(fileobj, *args, **kwargs)
    return forceopen_


class _FileWrapper(object):
    """
    This class provides a wrapper adding the option to force temporary reopen a
    file for I/O action in case they have been closed. This may be useful
    where files have been closed due to context managers, or inside functions

    forceopen: False reflects the built in open file. True forcing reopen


    Note:
    use forceopen=True carefully, you may modify any file even if closed.

    This class is not derived from file so as to make it easy to wrap this class
    around a already created fileobj with forcedfile(fileobj, ...)
    """
    def __init__(self, fileobj, forceopen=False):
        self.file = fileobj
        self.forceopen = forceopen

    def __getattr__(self, arg):
        try:
            return self.file.__getattribute__(arg)
        except AttributeError:
            try:
                return self.file.__getattr__(arg)
            except AttributeError:
                raise AttributeError(
                    self.__class__.__name__ + " object has no attribute '" +
                        arg + "'")

    def __str__(self):
        return 'forcedfile(' + self.file.__str__() + ')'

    def __repr__(self):
        return 'forcedfile(' + self.file.__repr__() + ')'

    @property
    def name(self):
        return self.file.name

    @property
    def mode(self):
        return self.file.mode

    @property
    def closed(self):
        return self.file.closed

    @property
    def encoding(self):
        return self.file.encoding

    @property
    def errors(self):
        return self.file.errors

    @property
    def newlines(self):
        return self.file.newlines

    @property
    def softspace(self):
        return self.file.softspace

    @property
    def forceopen(self):
        return self._forceopen

    @forceopen.setter
    def forceopen(self, v):
        if not isinstance(v, bool):
            raise ValueError('forceopen must be boolean')
        self._forceopen = v

    @forceopen_decorator
    def write(self, *args, **kwargs):
        return self.file.write(*args, **kwargs)

    @forceopen_decorator
    def __iter__(self):
        return self.file.__iter__()

    @forceopen_decorator
    def __enter__(self):
        self.file.__enter__()
        return self

    @forceopen_decorator
    def __exit__(self, exc, value, tb):
        result = self.file.__exit__(exc, value, tb)
        return result

    @forceopen_decorator
    def writelines(self, *args, **kwargs):
        return self.file.writelines(*args, **kwargs)

    @forceopen_decorator
    def writeable(self, *args, **kwargs):
        return self.file.writeable(*args, **kwargs)

    @forceopen_decorator
    def read(self, *args, **kwargs):
        return self.file.read(*args, **kwargs)

    @forceopen_decorator
    def readline(self, *args, **kwargs):
        return self.file.readline(*args, **kwargs)

    @forceopen_decorator
    def readlines(self, *args, **kwargs):
        return self.file.readlines(*args, **kwargs)

    @forceopen_decorator
    def xreadlines(self, *args, **kwargs):
        return self.file.xreadlines(*args, **kwargs)

    @forceopen_decorator
    def readall(self, *args, **kwargs):
        return self.file.readall(*args, **kwargs)

    @forceopen_decorator
    def readable(self, *args, **kwargs):
        return self.file.readable(*args, **kwargs)

    @forceopen_decorator
    def tell(self, *args, **kwargs):
        return self.file.tell(*args, **kwargs)

    @forceopen_decorator
    def seek(self, *args, **kwargs):
        return self.file.seek(*args, **kwargs)

    @forceopen_decorator
    def seekable(self, *args, **kwargs):
        return self.file.seekable(*args, **kwargs)

    @forceopen_decorator
    def flush(self, *args, **kwargs):
        return self.file.flush(*args, **kwargs)

    @forceopen_decorator
    def next(self, *args, **kwargs):
        return self.file.next(*args, **kwargs)

    @forceopen_decorator
    def fileno(self, *args, **kwargs):
        return self.file.fileno(*args, **kwargs)

    @forceopen_decorator
    def close(self, *args, **kwargs):
        return self.file.close(*args, **kwargs)

    @forceopen_decorator
    def isatty(self, *args, **kwargs):
        return self.file.isatty(*args, **kwargs)

    @forceopen_decorator
    def truncate(self, *args, **kwargs):
        return self.file.truncate(*args, **kwargs)


def isfileobj(obj):
    """ checks whether obj is a fileobject """
    return isinstance(obj, file) or isinstance(obj, _TemporaryFileWrapper) or isinstance(obj, _FileWrapper)


def getname(obj):
    """ returns path name of a fileobj or passes string through """
    if isinstance(obj, str):
        name = obj
    elif isfileobj(obj):
        name = obj.name
    else:
        raise TypeError('must be path or fileobject')
    return name


def getfileobj(obj, *args, **kwargs):
    """ returns fileobj from string or passes fileobj through"""
    if isinstance(obj, str):
        obj = open(obj, *args, **kwargs)
    elif not isfileobj(obj):
        raise TypeError('must be path or fileobject')
    return obj


def forcedfile(fileobj, forceopen=False):
    """
    returns the fileobject instance wraped such, that it may be forced open

    > fileobj: a fileobject as returned by open()
    > forceopen:
        True: will temporarily open an eventually closed file for I/O action
        False: no forced opening (built in fileobject unchanged)

    Examples:
    >>> import tempfile as tf
    >>> import os
    >>> fname = os.path.join(tf.gettempdir(),'test_filehandler.edf')
    >>> if os.path.exists(fname): os.remove(fname)

    >>> o = open(fname, mode='a+')  # just to hand over to xfile
    >>> xf = forcedfile(o, forceopen=False)
    >>> xf.write('abc')
    >>> xf.close()
    >>> xf.write('def')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...

    >>> xt = forcedfile(o, forceopen=True)
    >>> xt.write('ghi')
    >>> xt.close()
    >>> xt.write('jkl')
    >>> print xt.closed
    True

    >>> xr = forcedfile(o, forceopen=True)
    >>> xr.read()
    'abcghijkl'
    >>> print xt.closed
    True

    # a closed fileobj cannot enter a context manager
    >>> with forcedfile(o, forceopen=False) as xf:
    ...     xf.write('ABC')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...

    >>> with forcedfile(o, forceopen=True) as xt:
    ...     xt.write('GHI')
    ...     xt.close()
    ...     xt.write('JKL')  # doctest: +IGNORE_EXCEPTION_DETAIL

    >>> xr = forcedfile(o, forceopen=True)
    >>> xr.read()
    'abcghijklGHIJKL'
    >>> xr.close()

    >>> import os
    >>> os.remove(fname)
    """
    return _FileWrapper(fileobj, forceopen=forceopen)


def openforced(fname, forceopen=False, *args, **kwargs):
    """
    returns a newly created instance wraped such, that may be forced open

    > obj: a string containing the filepath
    > forceopen:
        True: will temporarily open an eventually closed file for I/O action
        False: no forced opening (built in fileobject unchanged)

    Examples:
    >>> import tempfile as tf
    >>> import os
    >>> fname = os.path.join(tf.gettempdir(),'test_filehandler.edf')
    >>> if os.path.exists(fname): os.remove(fname)

    >>> xf = openforced(fname, forceopen=False, mode='a+')
    >>> xf.write('abc')
    >>> xf.close()
    >>> xf.write('def')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...

    >>> xt = openforced(fname, forceopen=True, mode='a+')
    >>> xt.write('ghi')
    >>> xt.close()
    >>> xt.write('jkl')
    >>> print xt.closed
    True

    >>> xr = openforced(fname, forceopen=True, mode='a+')
    >>> xr.read()
    'abcghijkl'
    >>> print xt.closed
    True

    >>> with openforced(fname, forceopen=False, mode='a+') as xf:
    ...     xf.write('ABC')  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     xf.close()
    ...     xf.write('DEF')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...

    >>> with openforced(fname, forceopen=True, mode='a+') as xt:
    ...     xt.write('GHI')
    ...     xt.close()
    ...     xt.write('JKL')  # doctest: +IGNORE_EXCEPTION_DETAIL

    >>> xr = openforced(fname, forceopen=True)
    >>> xr.read()
    'abcghijklABCGHIJKL'
    >>> #xx = openforced(fname, mode='a+', forceopen=True)
    >>> #print xr
    >>> #print xx
    >>> xr.close()

    >>> import os
    >>> #os.remove(fname)
    """
    fileobj = open(fname, *args, **kwargs)
    f = forcedfile(fileobj, forceopen=forceopen)
    return f


if __name__ == '__main__':
    import doctest
    print 'doctest running'
    doctest.testmod()
    print 'doctest end'