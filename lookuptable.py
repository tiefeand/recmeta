#-------------------------------------------------------------------------------
# Name:        look up table
# Purpose:     extending the numpy.recarray by global and field specific meta
#              data and and providing a mutable version as well as a version
#              synchronised with a file content
#
# Author:      atiefenauer
#
# Created:     06.03.2014
# Copyright:   (c) atiefenauer 2014
#-------------------------------------------------------------------------------
#!/usr/bin/env python

__author__ = 'atiefenauer'

# ------------------------------------------------------------------------------
# DESCRIPTION:
# Lookuptables contain meta information that are bound to its recarray data.
# There are 3 different types (fieldmeta and globalmeta):
# - lookuptable (subtype of numpy.recarray)
# - mutablelut (mutable version of lookuptable)
# - filelut (mutable version that is coupled to a file)
#
# These factory functions are based on factory classed. Do not instantiate them
# directly
# - LookUpTable
# - MutbaleLookUpTable
# - FileLookUpTable
#
# the inherited interface of numpy.recarray is extended by additional methods
# (see '.index', '.exists', '.get', '.append', and '.record') that use **kwargs
# (resp. dictionaries) as a pythonic interface. Iterators such as one returning
# dictionaries is implemented too.
#
# run this module to test doctests
#
# reference to subclass recarray:
# http://docs.scipy.org/doc/numpy/reference/arrays.classes.html

# TODO: provide mutablelut() to create an empty instance instead of having to
# TODO: write mutablelut(None, shape=(), dtype=bool)

# ------------------------------------------------------------------------------

import numpy as np
recarray = np.recarray
format_parser = np.format_parser
rarray = np.rec.array
ndarray = np.ndarray
concatenate = np.concatenate
genfromtxt = np.genfromtxt
void = np.void
from matplotlib.mlab import rec2txt
import warnings
from abc import ABCMeta, abstractmethod
from copy import copy

from filehandler import isfileobj
from lutfilewriter import LutFileWriter, \
    DEFAULT_DELIMITER, DEFAULT_KEYVALUE_ASSIGNER, DEFAULT_KEYVALUE_SEPARATOR



DEFAULT_COMMENTS = '#'


def globalmeta_parser(kvpairs=None):
    try:
        globalmeta = dict() if kvpairs is None else dict(kvpairs)
    except:
        raise Exception('could not parse globalmeta, must be key value pairs')
    return globalmeta


def fieldmeta_parser(names=None, kvpairs=None):
    if isinstance(kvpairs, dict):
        if not all([k in names for k in kvpairs.keys()]):
            raise ValueError('keys must be a valid name: {0}'.format(names))
        return tuple([dict(kvpairs[n])
                      if n in kvpairs else dict() for n in names])
    else:
        try:
            if names:
                if kvpairs:
                    return tuple([dict(kvpairs[ii])
                                  for ii in range(len(names))])
                else:
                    return tuple([dict() for ii in range(len(names))])
            else:
                if kvpairs:
                    return tuple([dict(kv) for kv in kvpairs])
                else:
                    return tuple()
        except:
            raise ValueError('wrong type {0}. must be an iterable of key '.format(kvpairs) +
                             'value pairs of length {0}'.format(len(names)))


class LookUpTable(recarray):
    """ This is a factory class. Do not create instances by using this class.
        Use lookuptable() instead.

        Lookuptables expose the entire interface of numpy.recarray and add the
        possibility to additionally add global and/or field specific meta
        information. More convenience methods are added providing a pythonic
        dictionary like interface.

        As for recarray the datacontainer is immutable, but meta information
        may be altered at any time.

        Parameters
        ----------
        globalmeta : mapping (key value pair), optional
            any information that is valid on a global scope for this data set
        fieldmeta : iterable (list, tuple, dict) of mappings, optional
            any information that is valid for one particular column
        **kwargs :
            forwarded to the __new__ constructor of the super class recarray

        Returns
        -------
        lookuptable of the given shape, type and globalmeta, fieldmeta attribute

        See also
        --------
        numpy.recarray(), lookuptable()
    """
    def __new__(cls, globalmeta=None, fieldmeta=None, **kwargs):
        obj = recarray.__new__(cls, **kwargs)
        obj.globalmeta = globalmeta
        obj.fieldmeta = fieldmeta
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.globalmeta = getattr(obj, 'globalmeta', None)
        self.fieldmeta = getattr(obj, 'fieldmeta', None)

    def __repr__(self):
        repr_ = list()
        repr_.append('{data}'.format(data=self.tolist()))
        repr_.append('dtype={0}'.format(self.dtype))
        repr_.append('globalmeta={0}'.format(self.globalmeta)) \
            if self.globalmeta else ''
        repr_.append('fieldmeta={0}'.format(self.fieldmeta)) \
            if any([bool(f) for f in self.fieldmeta]) else ''
        repr_ = self.__class__.__name__ + '(' + ',\n\t'.join(repr_) + ')'
        return repr_.replace(self.__class__.__name__, 'lookuptable', 1)

    def __str__(self):
        str_ = list()
        if self.globalmeta:
            g = '\r\n'.join(['globalmeta:'] +
                ['\t{0}:{1}'.format(k, v) for k, v in self.globalmeta.items()])
            str_.append(g)
        if any([bool(fmeta) for fmeta in self.fieldmeta]) :
            fieldmeta = zip(*(self.names, self.fieldmeta))
            f = '\r\n'.join(['fieldmeta:'] +
                ['\t{0}:{1}'.format(n, f) for n, f in fieldmeta])
            str_.append(f)
        try:
            d = rec2txt(self.view(recarray), padding=3, precision=4)
        except:
            d = super(LookUpTable, self).__str__()
        str_.append(d)
        return '\r\n'.join(str_)

    def __nonzero__(self):
        return bool(self.nrecords)

    def _get_globalmeta(self):
        return self._globalmeta

    def _set_globalmeta(self, kvpairs=None):
        self._globalmeta = globalmeta_parser(kvpairs)

    globalmeta = property(_get_globalmeta, _set_globalmeta)

    def _get_fieldmeta(self):
        return self._fieldmeta

    def _set_fieldmeta(self, kvpairs=None):
        self._fieldmeta = fieldmeta_parser(self.dtype.names, kvpairs)

    fieldmeta = property(_get_fieldmeta, _set_fieldmeta)

    @property
    def names(self):
        return self.dtype.names

    @property
    def nfields(self):
        """ same as len(self.dtype) but returns 0 if object is unsized"""
        try:
            return len(self.dtype)
        except:
            return 0

    @property
    def nrecords(self):
        """ same as len(self) but returns 0 if object is unsized"""
        try:
            return len(self)
        except:
            return 0

    def iterrows(self):
        """ Returns an iterator object over records that yields records"""
        return (ii for ii in self)  # generator

    def iterfields(self):
        """ Returns an iterator object over fields that yields ndarrays"""
        return (self[ff] for ff in self.names)  # generator

    def iterdicts(self):
        """ Returns an iterator object over records that yields dictionaries"""
        return (dict(zip(*(self.names, ii))) for ii in self)  # generator

    def iterflat(self):
        """ Returns a flat iterator over the items"""
        return self.flat

    def iterrecords(self):
        return self.iterrows()

    def fielddict(self):
        """ Returns a dictionary with fieldnames and the corresponding fielddata as ndarrays"""
        try:
            return {n: self[n] for n in self.names}
        except:
            return dict()

    def keys(self):
        return self.names

    def index(self, **kwargs):
        """ Returns first record index that matches the keyword argument.

            Parameters
            ----------
            > **kwargs: a single key word argument where key is a fieldname and
                the value is the item to match.

            Error Handling
            --------------
            - AttributeError if data array is empty (sum(self.shape) == 0)
            - KeyError if key is not an existing fieldname
            - ValueError if value does not exist in field
        """
        if len(kwargs) != 1:
            clsname = self.__class__.__name__ + '.index()'
            raise AttributeError(
                'wrong amount of keyword arguments. {clsname} needs exactly ' +
                'one keyword argument'.format(
                    clsname=clsname))
        field = self.field(kwargs.keys()[0])  # KeyError if key not exists
        return field.tolist().index(
            kwargs.values()[0])  # ValueError if value does not exist

    def item(self, *args):
        """ Returns item or entire record respectively. This function may be
            used in two ways:

            One Parameters
            --------------
            > *args: a single integer defining the index of the RECORD to return

            Error Handling
            --------------
            - IndexError if data array is empty (sum(self.shape) == 0) or if
                given index is too high (or has wrong type)

            Two Parameters
            --------------
            > *args: an index and a fieldname of the ITEM to be returned.
                the fieldindex may be used instead of the name.

            Error Handling
            --------------
            - IndexError if data array is empty (sum(self.shape) == 0) or if
                index is too high
            - KeyError if fieldname does not exist
            - ValueError if fieldname exists but index has wrong type
        """
        try:
            return super(LookUpTable, self).item(*args)  # may raise IndexError
        except (ValueError, TypeError):  # TypeError: single non-int argument,
                                         # ValueError: is raised if 2 arguments
            try:
                return self.field(args[1])[args[0]]  # may raise IndexError
            except:
                raise

    def record(self, index):
        """ Returns the record at index

            Parameters
            ----------
            > index: record index

            Error Handling
            --------------
            - IndexError if index too high
        """
        if isinstance(index, int):
            return dict(zip(*(self.names, self[index])))
        else:
            raise TypeError(
                '\'{ix}\' not allowed, index must be integer'.format(ix=index))

    def exists(self, **kwargs):
        """ Checks whether an item in a particular field exists. Returns boolean

            Parameters
            ----------
            > **kwargs: a single key word argument where key is a fieldname and
                the value is the item to match.
        """
        try:
            self.index(**kwargs)
            return True
        except (ValueError, KeyError, IndexError):
            return False

    def get(self, field, **kwargs):
        """ Returns the item of a particular 'field' of the first record that
            matches the keyword argument

            Parameters
            ----------
            > field: fieldname of which an item is to be extracted
            > **kwargs: a single key word argument where key is a fieldname and
                the value is the item to match.

            Error Handling
            --------------
            - AttributeError if data array is empty (sum(self.shape) == 0)
            - KeyError if field or key of kwargs is not an existing fieldname
            - ValueError if value does not exist in field
        """
        index = self.index(**kwargs)
        return self.item(index, field)


def lookuptable(obj, fieldmeta=None, globalmeta=None, **kwargs):  # implement a way to pass subclasses through,... see subok in np.array
    """ Returns a LookUpTable object created out of a wide variety of objects.

        Based on numpy.rec.array it accepts any obj that is also handeled in
        numpy.rec.array(). In addition it also accepts:
        - dictionaries of scalars or iterables
        - list of dictionaries
        - other lookuptable, mutablelut or filelut objects
        - file objects pointing to a csv file as created with fileluts

        globalmeta and fieldmeta may be set providing them as arguments, any other
        optional keyword arguments are passed on to numpy.rec.array.

        See also
        --------
        numpy.rec.array()

        Examples
        --------
        >>> lut0 = lookuptable(None, shape=(), dtype=bool)
        >>> bool(lut0)
        False
        >>> lut0.fielddict()
        {}
        >>> lut0  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        lookuptable(...,
            dtype=|V1)
        >>> lookuptable(dict(a=[1,2,3], b=[4,5]), dtype=[('a','f4'),('b','i4')])  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 4), (2.0, 5)],
            dtype=[('a', '<f4'), ('b', '<i4')])
        >>> lookuptable(dict(a=1, b=2), formats=['f4', 'i4'])  # doctest: +NORMALIZE_WHITESPACE
        lookuptable((1.0, 2),
            dtype=[('a', '<f4'), ('b', '<i4')])
        >>> listofdicts = [dict(a=1, b=2),dict(b=4, a=3)]
        >>> lookuptable(listofdicts, names='a,b', formats=['f4', 'i4'])  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('a', '<f4'), ('b', '<i4')])
        >>> lookuptable(listofdicts, names='b,a')  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(2, 1), (4, 3)],
            dtype=[('b', '<i4'), ('a', '<i4')])
        >>> lookuptable(listofdicts, names='a,x')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ...
        >>> lookuptable([[1,2],[3,4]], dtype=[('a','f4'),('b','i4')])  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('a', '<f4'), ('b', '<i4')])
        >>> lut = lookuptable([[1,2],[3,4]], dtype=[('a','f4'),('b','i4')])
        >>> lookuptable(lut)  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('a', '<f4'), ('b', '<i4')])
        >>> lut.globalmeta['Version'] = 1
        >>> lut.fieldmeta = {'b': {'Unit':'mm'}}
        >>> lut  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('a', '<f4'), ('b', '<i4')],
            globalmeta={'Version': 1},
            fieldmeta=({}, {'Unit': 'mm'}))
        >>> lookuptable(None, shape=(), formats='f4,f4', names=('a1','a2'),
        ...     globalmeta={'Date': '2099-01-01', 'Version': 3.0},
        ...     fieldmeta={'a1':{'Unit':'mm'},'a2':{'Unit':'cm'}})  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        lookuptable(...,
            dtype=[('a1', '<f4'), ('a2', '<f4')],
            globalmeta={'Date': '2099-01-01', 'Version': 3.0},
            fieldmeta=({'Unit': 'mm'}, {'Unit': 'cm'}))
        >>> lut  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('a', '<f4'), ('b', '<i4')],
            globalmeta={'Version': 1},
            fieldmeta=({}, {'Unit': 'mm'}))
        >>> lut.view(LookUpTable)  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('a', '<f4'), ('b', '<i4')],
            globalmeta={'Version': 1},
            fieldmeta=({}, {'Unit': 'mm'}))
        >>> lut.view(type=LookUpTable, dtype=[('c','f4'),('d','i4')])  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('c', '<f4'), ('d', '<i4')],
            globalmeta={'Version': 1},
            fieldmeta=({}, {'Unit': 'mm'}))
        >>> print lut  # doctest: +NORMALIZE_WHITESPACE
        globalmeta:
            Version:1
        fieldmeta:
            a:{}
            b:{'Unit': 'mm'}
                a   b
           1.0000   2
           3.0000   4
        >>> #dict(lut)
        >>> for record in lut.iterrows(): record
        (1.0, 2)
        (3.0, 4)
        >>> for field in lut.iterfields(): field
        array([ 1.,  3.], dtype=float32)
        array([2, 4])
        >>> for record in lut.iterdicts(): record
        {'a': 1.0, 'b': 2}
        {'a': 3.0, 'b': 4}
        >>> for item in lut.iterflat(): item
        (1.0, 2)
        (3.0, 4)
        >>> lut.fielddict()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        {'a': array([ 1.,  3.], ...), 'b': array([2, 4])}
        >>> lut.record(1)
        {'a': 3.0, 'b': 4}
        >>> lut.index(b=4)
        1
        >>> lut.item(1)
        (3.0, 4)
        >>> lut.exists(a=3)
        True
        >>> lut.get('a', b=4)
        3.0
        >>> lut.globalmeta = None
        >>> lut.fieldmeta = None
        >>> lut  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1.0, 2), (3.0, 4)],
            dtype=[('a', '<f4'), ('b', '<i4')])
    """
    if isinstance(obj, (list, tuple)) \
            and isinstance(obj[0], dict):  # [{...},{...}] -> {[...],[...]}
        obj = dict([(n, [d[n] for d in obj])
                          for n in obj[0].keys()])

    if isinstance(obj, dict):  # {[...],[...]} -> [[...],[...]]
        key, val = zip(*obj.items())  # simult. unpacking to ensure order
        keynames = list(map(str, key))
        if kwargs.has_key('names') and kwargs.get('names'):  # kwargs.get('names') must not be None
            names = kwargs.get('names')
            if isinstance(names, str):
                names = names.split(',')
            if not set(names) == set(keynames):
                raise ValueError('names must be out of ' + ','.join(map(str, keynames)))
            val = [obj[n] for n in names]
        else:
            kwargs.update(names=keynames)
        if all(map(lambda v: isinstance(v, (list, tuple)), val)):
            obj = zip(*val)
        else:  # if dictionary values are scalars
            obj = val

    if isinstance(obj, AbstractMutableLookUpTable):
        obj = obj.tolut()

    elif isfileobj(obj):
        obj = lutfromcsv(obj)

    # makes sure to use names even if formats not explicitely defined (as it
    # would be the case in the underlying np.rec.array())
    if kwargs.has_key('names') and kwargs.get('names'):  # kwargs.get('names') must not be None
        names = kwargs.get('names')
        if isinstance(names, str):
            names = names.split(',')
        names = list(names)
        kwargs.update({'names': names})
        if not kwargs.has_key('formats'):
            tmp = rarray(obj, copy=False)
            formats = list(zip(*tmp.dtype.descr)[1])
            kwargs.update({'formats': formats})

    a = rarray(obj, **kwargs).view(LookUpTable)  # always copies data

    # meta is taken from obj if any, otherwise overwritten with arguments
    # 'globalmeta', 'fieldmeta' (resp.) if any.
    if globalmeta:
        a.globalmeta = globalmeta
    if fieldmeta:
        a.fieldmeta = fieldmeta
    return a


array = lookuptable


# ******************************************************************************
def _recfromcsv(fname, **kwargs):
    """ Reimplementation of npyio.recfromcsv with bug fix
        - now: correct updating of dtype
        - no transformation of names with case_sensitive
    """
    dtype = None
    case_sensitive = kwargs.get('case_sensitive', "lower") or "lower"
    names = kwargs.get('names', True)
    if names is None:
        names = True
    kwargs.update(dtype=kwargs.get('dtype', dtype),
                  delimiter=kwargs.get('delimiter', ",") or ",",
                  names=names,
                  case_sensitive=case_sensitive)
    usemask = kwargs.get("usemask", False)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output
# ******************************************************************************


def lutfromcsv(fid, **kwargs):
    """ Reads lookuptable from a csv file

        The file may consist of the following sections
        globalmeta: lines with a key-value pair but nothing else
        fieldmeta: lines with at least a key-value pair and a key-value separator
            or a delimiter
        names: a line with names
        data: spreadsheet like data separated by delimiters

        format of file content (example)
        ----------------------
        Date=2014-01-23
        Version=3.0
        Unit=None    Unit=km,Nr=3  Unit=cm,Nr=7
        Sample  SensorA SensorB
        1   3.21    5.22
        2   4.31    6.35

        Examples
        --------
        >>> #import tempfile as tf
        >>> #import os
        >>> #tempfile = os.path.join(tf.gettempdir(),'test_lutfromcsv.csv')
        >>> #if os.path.exists(tempfile): os.remove(tempfile)
        >>> import tempfile as tf
        >>> tempfile = tf.TemporaryFile()
        >>> flut = filelut(tempfile, shape=(),
        ...     formats='i4,f4,f4',
        ...     names=['Sample', 'SensorA', 'SensorB'])
        >>> flut.globalmeta = {'Date': '2014-01-23', 'Version': 3.0}
        >>> flut.fieldmeta = {'Sample': {'Unit':None},
        ...                   'SensorA': {'Unit':'km','Nr':3},
        ...                   'SensorB': {'Unit':'cm','Nr':7}}
        >>> flut.append(dict(Sample=1, SensorA=3.21, SensorB=5.22))
        >>> flut.append(dict(SensorA=4.31, Sample=2, SensorB=6.35))
        >>> flut = lutfromcsv(tempfile)
        >>> flut  # doctest: +NORMALIZE_WHITESPACE
        lookuptable([(1, 3.21, 5.22), (2, 4.31, 6.35)],
            dtype=[('Sample', '<i4'), ('SensorA', '<f8'), ('SensorB', '<f8')],
            globalmeta={'Date': '2014-01-23', 'Version': '3.0'},
            fieldmeta=({'Unit': 'None'}, {'Nr': '3', 'Unit': 'km'}, {'Nr': '7', 'Unit': 'cm'}))

    """
    delim = DEFAULT_DELIMITER
    separ = DEFAULT_KEYVALUE_SEPARATOR
    assgn = DEFAULT_KEYVALUE_ASSIGNER
    flag_close = False
#    if isfileobj(fid):
#        fid = fid.name
    if isinstance(fid, str):
        fid = open(fid, 'a+')
        flag_close = True
    start_header = fid.tell()
    fid.seek(0)
    # if (offset > 0):
    #     fid.seek(offset, 1)
    # start = fid.tell()

    # parsing header consisting of whitespace or assgn
    nheader = 0
    globalmeta = dict()
    fieldmeta = []
    line = fid.readline()
    while line:
        if assgn in line:
            # detect ncol before striped since delimiter may be a whitespace
            ncol = len(line.split(delim))
            cols = line.strip().split(delim)
            if ncol > 1 or (separ in line and '{' not in line and '[' not in line):  # a fieldmeta line detected
                cols = [col.strip(separ).split(separ) for col in cols]
                cols = [[ii.split(assgn) for ii in col] for col in cols]
                for ii, col in enumerate(cols):
                    try:
                        fieldmeta[ii].update(col)
                    except IndexError:
                        fieldmeta.append(dict(col))
            else:  # a globalmeta line detected
                globalmeta.update([cols[0].split(assgn)])
        elif line.strip():  # neither meta data line nor a whitespace-line
            break
        nheader += 1
        line = fid.readline()

    kwargs.update(delimiter=kwargs.get('delimiter', delim),
                  comments=kwargs.get('comments', DEFAULT_COMMENTS),
                  names=kwargs.get('names', True),
                  skip_header=kwargs.get('skip_header', nheader),
                  case_sensitive=True,
                  replace_space='_'
                  )
    # start_csv = fid.tell()
    fid.seek(0)  # eventually to be replaced with fid.seek(start_header)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            rec = _recfromcsv(fname=fid, **kwargs)
        except (IndexError, ValueError):
            raise ValueError('reading of csv part failed (data section empty).')

    fid.seek(start_header)
    if flag_close:
        fid.close()
    return lookuptable(rec, globalmeta=globalmeta, fieldmeta=fieldmeta)


class AbstractMutableLookUpTable(object):
    """ This is a factory class. Do not create instances by using this class.
        Use mutablelut() instead.

        MutableLookUpTables are composed of lookuptables but allow to update
        their content by adding records. Thus a lookuptable may be intialized
        empty but can grow.

        Returns
        -------
        mutablelut of the given shape, type and globalmeta, fieldmeta attribute

        See also
        --------
        LookUpTable(), mutablelut()
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def fromarray(obj, **kwargs):
        """ HAS TO BE IMPLEMENTED

            This method needs to return any arraylike object that can be
            interpreted by lookuptable(). It acts as a transformator for data
            that is given just before instantiation of an object and also when
            content is appended using method .append().
        """
        raise NotImplementedError

    @classmethod
    def create(cls, obj, **kwargs):
        lut = cls.fromarray(obj, **kwargs)
        mlut = cls(shape=lut.shape,
            dtype=lut.dtype,
            buf=lut.data,
            globalmeta=lut.globalmeta,
            fieldmeta=lut.fieldmeta
            )
        return mlut

    def __init__(self, **kwargs):
        self._data = LookUpTable(**kwargs)

    def __repr__(self):
        return self._data.__repr__().replace('lookuptable', 'mutablelut', 1)

    def __str__(self):
        s = self._data.__str__()
        if self.isempty():
            s = s.strip().split('\r\n')
            s[-1] = '<empty>'
            s = '\r\n'.join(s)
        return s

    def __getattr__(self, arg):
        try:
            return self._data.__getattribute__(arg)
        except AttributeError:
            try:
                return self._data.__getattr__(arg)
            except AttributeError:
                raise AttributeError(
                    self.__class__.__name__ + " object has no attribute '" +
                        arg + "'")

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, *arg):
        return self._data.__getitem__(*arg)

    def __len__(self):
        if not self.isempty():
            return self._data.__len__()
        else:  # prevent exception saying TypeError: len() of unsized object
            return 0

    def __nonzero__(self):
        return bool(self._data)

    def __call__(self, model=None, **kwargs):
        dict_ = self.fielddict()
        dict_.update(**kwargs)
        if model:
            return model(**dict_)
        else:
            return dict_

    def _get_globalmeta(self):
        return self._data._get_globalmeta()

    def _set_globalmeta(self, kvpairs):
        self._data._set_globalmeta(kvpairs)

    globalmeta = property(_get_globalmeta, _set_globalmeta)

    def _get_fieldmeta(self):
        return self._data._get_fieldmeta()

    def _set_fieldmeta(self, kwargs):
        self._data._set_fieldmeta(kwargs)

    fieldmeta = property(_get_fieldmeta, _set_fieldmeta)

    def isempty(self):
        """ Returns boolean whether the mutablelut is empty. A mlut is defined
            empty if sum(shape) is 0, no matter what the assoziated content of
            the buffer is
        """
        return not bool(sum(self.shape))

    def tolut(self):
        """ Converts this instance to a LookUpTable object (immutable type)
        """
        return self._data.copy()

    @property
    def empty(self):
        return self.isempty()

    @property
    def nfields(self):
        return 0 if self.empty else self._data.nfields

    @property
    def nrecords(self):
        return 0 if self.empty else self._data.nrecords

    @property
    def names(self):
        return self._data.names

    def copy(self, **kwargs):
        return copy(self, **kwargs)

    def append(self, obj=None, updateglobalmeta=False, updatefieldmeta=False, casting='unsafe', passback=False, **kwargs):
        """ Appends any arraylike object. obj may be any type that is accepted
            in lookuptable() as well.

            To fit the existing array obj must come with the same names. Objects
            that have such defined are either subtyoes of recarray (lookuptable)
            mutablelut, dictionaries or list of dictionaries.

            obj, that do not carry names (such as tuples, lists, ...), will have
            to be accompagnied with the argument 'names' defining the order how
            the appended data is mapped to the existing data.

            In case obj carries fieldmeta or globalmeta data these may be added
            to the current array by choosing updateglobalmeta=True or
            updatefieldmeta=True respectively. By default they are dropped.

            Many more parameters may be provided squeezing the input data into a
            desired form.

            See also
            --------
            mutablelut(), lookuptable(), numpy.rec.array()
        """
        if obj is None:
            return
        arr = self.fromarray(obj, **kwargs)

        lut = lookuptable(arr)
        if self.names:
            if not set(lut.names) == set(self.names):
                raise TypeError(
                    ''.join(['The records\n{rows}\ncannot be added, names are ',
                    'either not defined or do not match the data:\n{data}']
                    ).format(rows=str(lut), data=str(self._data)))
            add = lut[list(self.names)]
            add = add.astype(self.dtype, casting=casting).view(LookUpTable)
        else:
            add = lut.copy()

        if not self.empty:
            new = concatenate((self._data, add))
            new = new.view(LookUpTable)
        else:
            new = add

        new.globalmeta = self.globalmeta
        new.fieldmeta = self.fieldmeta

        if updateglobalmeta:
            new.globalmeta.update(lut.globalmeta)
        if updatefieldmeta:
            idx_names = [list(lut.names).index(v) for v in new.names]
            [new.fieldmeta[i].update(lut.fieldmeta[j])
                for i, j in enumerate(idx_names)]
        super(AbstractMutableLookUpTable, self).__setattr__('_data', new)
        if passback:
            return add


class MutableLookUpTable(AbstractMutableLookUpTable):
    @staticmethod
    def fromarray(obj, **kwargs):
        """ Returns a MutableLookUpTable object created out of a wide variety of
            objects.

            Based on lookuptables it accepts any obj that is also handeled in
            lookuptable().

            Empty mutableluts may be defined without specifying names and formats
            (i.e. by dtype=bool) or alternatively by specifying them. In case they
            are defined, they will define the format of the subsequently growing
            array.

            >>> mlut1 = mutablelut(None, shape=(), formats='f4,f4', names='a,b',
            ...    fieldmeta=dict(a=dict(x=1)))
            >>> mlut1.isempty()
            True
            >>> bool(mlut1)
            False
            >>> mlut1.append(dict(b=2.5, a=1.1), formats=['i4', 'f4'])
            >>> mlut1  # doctest: +NORMALIZE_WHITESPACE
            mutablelut([(1.0, 2.5)],
                dtype=[('a', '<f4'), ('b', '<f4')],
                fieldmeta=({'x': 1}, {}))
            >>> bool(mlut1)
            True

            In the case of no specified names and formats, ( thus i.e. defined
            by dtype=bool), the first data appended will then define the arrays
            format and names. Here fieldmetas cannot yet be added because there are
            no defined fields yet (thus there are no names yet).

            >>> mlut2 = mutablelut(None, shape=(), dtype=bool)
            >>> mlut2.isempty()
            True
            >>> mlut2.append(dict(a=1.1, b=2.5), formats=['i4', 'f4'])
            >>> mlut2  # doctest: +NORMALIZE_WHITESPACE
            mutablelut([(1, 2.5)],
                dtype=[('a', '<i4'), ('b', '<f4')])

            The latter way of instantiation is useful in cases where the data, its
            names and/or formats are not yet defined at the point of instantiation.

            Note, that any data append to a non-empty mutablelut will be (down-)
            casted to the type in this mutablelut. I.e. pay attention with strings.
            As they are defined by length of letters, appending a longer string may
            truncate it to the length of the mutablelut.

            See also
            --------
            lookuptable(), numpy.rec.array()

            Examples
            --------
            >>> mlut2.append()  # nothing will be added
            >>> mlut2.append(dict(a=3.3,b=0.75))
            >>> mlut2.append([[4.6, 1.25],[5.3, 2.5]], names=('a','b'))
            >>> mlut2.append(dict(x=9,y=9))  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            TypeError: ...
            >>> mlut2.fieldmeta={'a':{'Unit':'cm'}}
            >>> mlut2.globalmeta.update({'Version': 3})
            >>> mlut2  # doctest: +NORMALIZE_WHITESPACE
            mutablelut([(1, 2.5), (3, 0.75), (4, 1.25), (5, 2.5)],
                dtype=[('a', '<i4'), ('b', '<f4')],
                globalmeta={'Version': 3},
                fieldmeta=({'Unit': 'cm'}, {}))
            >>> mlut2.isempty()
            False
            >>> f = dict(b={'Age':7}, a={'Age':3})
            >>> g = dict(Season='winter')
            >>> lut = lookuptable([8.5,7.7], names=('b','a'), fieldmeta=f, globalmeta=g)
            >>> mlut2.append(lut, updateglobalmeta=True, updatefieldmeta=True)
            >>> mlut2  # doctest: +NORMALIZE_WHITESPACE
            mutablelut([(1, 2.5), (3, 0.75), (4, 1.25), (5, 2.5), (7, 8.5)],
                dtype=[('a', '<i4'), ('b', '<f4')],
                globalmeta={'Season': 'winter', 'Version': 3},
                fieldmeta=({'Age': 3, 'Unit': 'cm'}, {'Age': 7}))
            >>> mlut2()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            {'a': array([1, 3, 4, 5, 7]), 'b': array([ 2.5 ,  0.75,  1.25,  2.5 ,  8.5 ], ...)}
            >>> mlut2(**{'c': 9})  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            {'a': array([1, 3, 4, 5, 7]), 'c': 9, 'b': array([ 2.5 ,  0.75,  1.25,  2.5 ,  8.5 ], ...)}
            >>> def fun(a, b, c):
            ...     return a*b+c, a-b
            >>> mlut2(fun, **dict(c=2))  # doctest: +NORMALIZE_WHITESPACE,
            (array([  4.5 ,   4.25,   7.  ,  14.5 ,  61.5 ]), array([-1.5 ,  2.25,  2.75,  2.5 , -1.5 ]))
        """
        lut = lookuptable(obj, **kwargs)
        # if obj is None and lut.names:
        #     lut = lookuptable({n: np.array([]) for n in lut.names})
        if obj is not None and not lut.shape:
            lut = lut.reshape(1)
        return lut


mutablelut = MutableLookUpTable.create


class AbstractFileLookUpTable(AbstractMutableLookUpTable):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def fromfile(fid, **kwargs):
        """ HAS TO BE IMPLEMENTED

            This method needs to return any arraylike object that can be
            interpreted by lookuptable(). It acts as a transformator for data
            that is given upon instantiation of an object.
        """
        raise NotImplementedError

    @classmethod
    def create(cls, fid, **kwargs):
        try:
            mlut = cls.fromfile(fid, **kwargs)
            # flut = cls(fid,
            #     shape=mlut.shape,
            #     dtype=mlut.dtype,
            #     buf=mlut.data,
            #     globalmeta=mlut.globalmeta,
            #     fieldmeta=mlut.fieldmeta
            #     )
        except ValueError:
            mlut = cls.fromfile(None, **kwargs)
        flut = cls(fid,
            shape=mlut.shape,
            dtype=mlut.dtype,
            buf=mlut.data,
            globalmeta=mlut.globalmeta,
            fieldmeta=mlut.fieldmeta
            )
        return flut

    def __init__(self, fileobj, **kwargs):
        super(AbstractFileLookUpTable, self).__init__(**kwargs)
        self.file = fileobj

    def __repr__(self):
        fileobj = self.file
        return self.__class__.__name__ + '(\'{fileobj}\')'.format(fileobj=fileobj)

    def __str__(self):
        fname = self.file.name
        data = super(AbstractFileLookUpTable, self).__str__()
        return '\'{fname}\':\n{data}'.format(fname=fname, data=data)

    def __enter__(self):
        self.file.__enter__()
        return self

    def __exit__(self, exc, value, tb):
        result = self.file.__exit__(exc, value, tb)
        self.file.close()
        return result

    def _get_globalmeta(self):
        return super(AbstractFileLookUpTable, self)._get_globalmeta()

    def _set_globalmeta(self, globalmeta):
        if self.empty:
            super(AbstractFileLookUpTable, self)._set_globalmeta(globalmeta)
        else:
            raise IOError(
                'cannot add globalmeta if data section of file is not empty')

    globalmeta = property(_get_globalmeta, _set_globalmeta)

    def _get_fieldmeta(self):
        return super(AbstractFileLookUpTable, self)._get_fieldmeta()

    def _set_fieldmeta(self, fieldmeta):
        if self.empty:
            super(AbstractFileLookUpTable, self)._set_fieldmeta(fieldmeta)
        else:
            raise IOError(
                'cannot add fieldmeta if data section of file is not empty')

    fieldmeta = property(_get_fieldmeta, _set_fieldmeta)

    def append(self, obj=None, updateglobalmeta=False, updatefieldmeta=False, casting='unsafe', **kwargs):
        """ Appends any arraylike object. obj may be any type that is accepted
            by the append of the super class (see MutableLookUpTable.append()).

            If successfully appended data will be written to the file.

            See also
            --------
            MutableLookUpTable.append()
        """
        # check emptyness before appending, then append and write
        empty = self.empty
        out = super(AbstractFileLookUpTable, self).append(obj,
                                                          updatefieldmeta=updatefieldmeta,
                                                          updateglobalmeta=updateglobalmeta,
                                                          casting=casting,
                                                          **kwargs)
        # write only once append succeeded
        writer = LutFileWriter(self.file, self.names)
        if empty:
            writer.writeglobalmeta(self.globalmeta)
            writer.writefieldmeta(self.fieldmeta)
            writer.writeheader()
        writer.writerows(lookuptable(self.fromarray(obj, **kwargs)).iterdicts())
        return out


class FileLookUpTable(AbstractFileLookUpTable):

    @staticmethod
    def fromarray(obj, **kwargs):
        return mutablelut(obj, **kwargs)

    @staticmethod
    def fromfile(fid, **kwargs):
        """ Returns a FileLookUpTable object created given a file object.

            The File version of a mutablelut holds data in memory synchronised with
            the content of the file given with the fileobject fid.

            Examples
            --------
            >>> import tempfile as tf
            >>> tempfile = tf.TemporaryFile()
            >>> flut = filelut(tempfile, shape=(), formats='i4,f4,f4',
            ...     names=['Sample', 'SensorA', 'SensorB'])
            >>> flut.globalmeta = {'Date': '2014-01-23', 'Version': 3.0}
            >>> flut.fieldmeta = {'Sample': {'Unit':None},
            ...                   'SensorA': {'Unit':'km','Nr':3},
            ...                   'SensorB': {'Unit':'cm','Nr':7}}
            >>> flut.append(dict(Sample=1, SensorA=3.21, SensorB=5.22))
            >>> flut.append(dict(SensorA=4.31, Sample=2, SensorB=6.35))
            >>> flut.append(dict(X=1, SensorA=4.31, SensorB=6.35))  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            TypeError: ...

            >>> print flut  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            '...':
            globalmeta:
                Date:2014-01-23
                Version:3.0
            fieldmeta:
                Sample:{'Unit': None}
                SensorA:{'Nr': 3, 'Unit': 'km'}
                SensorB:{'Nr': 7, 'Unit': 'cm'}
               Sample   SensorA   SensorB
                    1    3.2100    5.2200
                    2    4.3100    6.3500
            >>> flut = filelut(tempfile)
            >>> flut  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            FileLookUpTable('<open file '<fdopen>', mode 'w+b' at ...>')
            >>> print flut  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            '...':
            globalmeta:
                Date:2014-01-23
                Version:3.0
            fieldmeta:
                Sample:{'Unit': 'None'}
                SensorA:{'Nr': '3', 'Unit': 'km'}
                SensorB:{'Nr': '7', 'Unit': 'cm'}
               Sample   SensorA   SensorB
                    1    3.2100    5.2200
                    2    4.3100    6.3500
            >>> flut.file.close()
            >>> f = {'Sample': {'Unit':None},
            ...      'SensorA': {'Unit':'km','Nr':3},
            ...      'SensorB': {'Unit':'cm','Nr':7}}
            >>> tempfile = tf.TemporaryFile()
            >>> with filelut(tempfile, shape=(), dtype=bool) as flut:
            ...     flut.globalmeta = {'Date': '2014-01-23', 'Version': 3.0}
            ...     flut.append(dict(Sample=1, SensorA=3.21, SensorB=5.22), fieldmeta=f, updatefieldmeta=True)
            ...     flut.append(dict(SensorA=4.31, Sample=2, SensorB=6.35))
            ...     print flut # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            ...     flut = filelut(tempfile)
            ...     print flut # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            '...':
            globalmeta:
                Date:2014-01-23
                Version:3.0
            fieldmeta:
                SensorA:{'Nr': 3, 'Unit': 'km'}
                Sample:{'Unit': None}
                SensorB:{'Nr': 7, 'Unit': 'cm'}
               SensorA   Sample   SensorB
                3.2100        1    5.2200
                4.3100        2    6.3500
            '...':
            globalmeta:
                Date:2014-01-23
                Version:3.0
            fieldmeta:
                SensorA:{'Nr': '3', 'Unit': 'km'}
                Sample:{'Unit': 'None'}
                SensorB:{'Nr': '7', 'Unit': 'cm'}
               SensorA   Sample   SensorB
                3.2100        1    5.2200
                4.3100        2    6.3500
        """
        return mutablelut(fid, **kwargs)  # mutablelut is able to interpret both
                                          # arraylike or filebj. but here could
                                          # also be placed a function that only
                                          # accepts file objects, while on the
                                          # other hand fromarray does not need
                                          # to be able read fileobj


filelut = FileLookUpTable.create


if __name__ == '__main__':

    # f = open(r'C:\Users\atiefenauer\Desktop\Meas\aaa.txt', mode='a+')
    # mutablelut(f)

    import doctest
    print 'doctest running'
    doctest.testmod()
    print 'doctest end'