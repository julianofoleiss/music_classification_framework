# -*- coding: utf-8 -*-
import tables
import six
import os
import tempfile

def create_earray(filename, data_type=tables.Float32Atom(), array_name='features', fixdim_size=10, compression_level=1, compression_lib='lzo', temp=False):

    output = filename

    if temp:
        if isinstance(temp, six.string_types):
            output = temp + output + '.h5'
        else:
            tf = tempfile.NamedTemporaryFile('w', delete=False)
            os.unlink(tf.name)
            output = tf.name

    a = data_type

    h5 = tables.open_file(output, mode='w')
    f = tables.Filters(complevel=compression_level, complib=compression_lib)
    outdata = h5.create_earray(h5.root, array_name, a, (0, fixdim_size), filters=f )
    outdata.attrs.h5filename = h5.filename

    return output, h5, outdata

def remove_earray(filename, h5obj):
    h5obj.flush()
    h5obj.close()

    os.unlink(filename)
