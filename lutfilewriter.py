#-------------------------------------------------------------------------------
# Name:        lutfilewriter
# Purpose:     wrapper around DictWriter adding the possibility to write
#              globalmeta and fieldmeta data
# Author:      atiefenauer
#
# Created:     15.05.2014
# Copyright:   (c) atiefenauer 2014
#-------------------------------------------------------------------------------
#!/usr/bin/env python

__author__ = 'atiefenauer'

# ------------------------------------------------------------------------------
# DESCRIPTION:
# ------------------------------------------------------------------------------
from csv import Dialect, register_dialect, DictWriter, DictReader, QUOTE_MINIMAL
from os import linesep
from filehandler import openforced

# csv.Dialect
DEFAULT_DELIMITER = '\t'
DEFAULT_DOUBLEQUOTE = True
DEFAULT_ESCAPECHAR = None
DEFAULT_LINEDETERMINATOR = linesep
DEFAULT_QUOTECHAR = '"'
DEFAULT_QUOTING = QUOTE_MINIMAL
DEFAULT_SKIPINITIALSPACE = False

# csv.DictWriter
DEFAULT_RESTVAL = ''
DEFAULT_EXTRASACTION = 'raise'
DEFAULT_DIALECT = 'filelut'
DEFAULT_RESTKEY = ['a']

# meta
DEFAULT_KEYVALUE_SEPARATOR = ','
DEFAULT_KEYVALUE_ASSIGNER = '='


class LutFileDialect(Dialect):
    delimiter = DEFAULT_DELIMITER
    doublequote = DEFAULT_DOUBLEQUOTE
    escapechar = DEFAULT_ESCAPECHAR
    lineterminator = DEFAULT_LINEDETERMINATOR
    quotechar = DEFAULT_QUOTECHAR
    quoting = DEFAULT_QUOTING
    skipinitialspace = DEFAULT_SKIPINITIALSPACE

register_dialect("filelut", LutFileDialect)


class LutFileWriter(DictWriter):
    """
    >>> import tempfile as tf
    >>> import os
    >>> fname = os.path.join(tf.gettempdir(),'test_lutdictwriter.txt')
    >>> if os.path.exists(fname): os.remove(fname)
    >>> xf = openforced(fname, mode='a+', forceopen=False)
    >>> l = LutFileWriter(xf, ['a','b','c'])
    >>> xf.close()
    >>> l.writefieldmeta([dict(x=1, u=4, v=5),dict(y=2),dict(z=3, w=6)])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...

    >>> xt = openforced(fname, mode='a+', forceopen=True)
    >>> l = LutFileWriter(xt, ['a','b','c'])
    >>> xt.close()
    >>> l.writeheader()  # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> l.writeglobalmeta(dict(e=1,f=2))
    >>> l.writerow(dict(a=1,b=2,c=3))  # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> l.writefieldmeta([dict(x=1, u=4, v=5),dict(y=2),dict(z=3, w=6)])  # doctest: +IGNORE_EXCEPTION_DETAIL

    >>> with openforced(fname, mode='a+', forceopen=False) as xf:
    ...     l = LutFileWriter(xf, ['a','b','c'])
    ...     l.writeheader()
    ...     l.writeglobalmeta(dict(g=1,h=2))
    ...     l.writerow(dict(a=4,b=5,c=6))
    ...     l.writefieldmeta([dict(x=1, u=4, v=5),dict(y=2),dict(z=3, w=6)])
    >>> l.writefieldmeta([dict(x=1, u=4, v=5), dict(y=2), dict(z=3, w=6)])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...

    >>> with openforced(fname, mode='a+', forceopen=True) as xt:
    ...     l = LutFileWriter(xt, ['a','b','c'])
    ...     l.writeheader()
    >>> l.writefieldmeta([dict(y=2),dict(x=1, u=4, v=5),dict(z=3, w=6)])  # doctest: +IGNORE_EXCEPTION_DETAIL

    >>> xf = openforced(fname, forceopen=False)
    >>> with xf:  # doctest: +NORMALIZE_WHITESPACE
    ...     for l in xf.readlines():
    ...         print l  # doctest: +NORMALIZE_WHITESPACE
    a	b	c
    <BLANKLINE>
    e=1
    <BLANKLINE>
    f=2
    <BLANKLINE>
    1	2	3
    <BLANKLINE>
    x=1,u=4,v=5,	y=2,	z=3,w=6,
    <BLANKLINE>
    a	b	c
    <BLANKLINE>
    h=2
    <BLANKLINE>
    g=1
    <BLANKLINE>
    4	5	6
    <BLANKLINE>
    x=1,u=4,v=5,	y=2,	z=3,w=6,
    <BLANKLINE>
    a	b	c
    <BLANKLINE>
    y=2,	x=1,u=4,v=5,	z=3,w=6,
    <BLANKLINE>
    >>> os.remove(fname)
    """
    def __init__(self, fileobj, fieldnames,
                 key_value_separator=DEFAULT_KEYVALUE_SEPARATOR,
                 key_value_assigner=DEFAULT_KEYVALUE_ASSIGNER, **kwargs):
        kwargs.update(restval=kwargs.get('restval', DEFAULT_RESTVAL),
                      extrasaction=kwargs.get('extrasaction', DEFAULT_EXTRASACTION),
                      dialect=kwargs.get('dialect', DEFAULT_DIALECT))
        self.fileobj = fileobj
        DictWriter.__init__(self, self.fileobj, fieldnames, **kwargs)
        self.key_value_separator = key_value_separator
        self.key_value_assigner = key_value_assigner

    def writeglobalmeta(self, dict_, **kwargs):
        assgn = self.key_value_assigner
        termi = self.writer.dialect.lineterminator
        [self.fileobj.write(assgn.join(map(str, [key, val])) + termi, **kwargs)
            for key, val in dict_.items()]

    def writefieldmeta(self, listofdicts, **kwargs):
        separ = self.key_value_separator
        assgn = self.key_value_assigner
        flist = []
        for f in listofdicts:
            flist.append([assgn.join(map(str, kv)) for kv in f.items()])
        flist = [separ.join(ff+['']) for ff in flist]
        fdict = dict(zip(*(self.fieldnames, flist )))
        if any([bool(f) for f in listofdicts]):
            self.writerow(fdict, **kwargs)


class LutFileReader(DictReader):
    """
    >>> import tempfile as tf
    >>> import os
    >>> fname = os.path.join(tf.gettempdir(),'test_lutdictreader.txt')
    >>> if os.path.exists(fname): os.remove(fname)
    >>> xt = openforced(fname, mode='a+', forceopen=True)
    >>> l = LutFileWriter(xt, ['a','b','c'])
    >>> xt.close()
    >>> l.writeheader()  # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> l.writeglobalmeta(dict(e=1,f=2))
    >>> l.writerow(dict(a=1,b=2,c=3))  # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> l.writefieldmeta([dict(x=1, u=4, v=5),dict(y=2),dict(z=3, w=6)])  # doctest: +IGNORE_EXCEPTION_DETAIL

    >>> xt = openforced(fname, mode='a+', forceopen=True)
    >>> r = LutFileReader(xt, ['a','b','c'])
    >>> r.next()  # doctest: +IGNORE_EXCEPTION_DETAIL
    {'a': 'a', 'c': 'c', 'b': 'b'}
    >>> r.readglobalmeta()
    {'e': '1'}
    >>> r.readglobalmeta()
    {'f': '2'}
    >>> r.next()  # doctest: +IGNORE_EXCEPTION_DETAIL
    {'a': '1', 'c': '3', 'b': '2'}
    >>> r.readfieldmeta()  # doctest: +IGNORE_EXCEPTION_DETAIL
    {'a': {'x': '1', 'u': '4', 'v': '5'}, 'c': {'z': '3', 'w': '6'}, 'b': {'y': '2'}}
    >>> xt.close()
    >>> os.remove(fname)
    """

    def __init__(self, fileobj, fieldnames=None,
                 key_value_separator=DEFAULT_KEYVALUE_SEPARATOR,
                 key_value_assigner=DEFAULT_KEYVALUE_ASSIGNER, **kwargs):
        kwargs.update(restval=kwargs.get('restval', DEFAULT_RESTVAL),
                      restkey=kwargs.get('restkey', DEFAULT_RESTKEY),
                      dialect=kwargs.get('dialect', DEFAULT_DIALECT))
        self.fileobj = fileobj
        DictReader.__init__(self, fileobj, fieldnames=fieldnames, **kwargs)
        self.key_value_separator = key_value_separator
        self.key_value_assigner = key_value_assigner

    def readglobalmeta(self):
        assgn = self.key_value_assigner
#        termi = self.reader.dialect.lineterminator
        return dict([self.next()[self.fieldnames[0]].split(assgn)])


    def readfieldmeta(self):
        separ = self.key_value_separator
        assgn = self.key_value_assigner
        items = self.next().items()
        keys, values = zip(*items)
        flist = [dict([kv.split(assgn) for kv in v.strip(separ).split(separ)]) for v in values]
        return dict(zip(*(keys, flist)))


if __name__ == '__main__':
    import doctest
    print 'doctest running'
    doctest.testmod()
    print 'doctest end'