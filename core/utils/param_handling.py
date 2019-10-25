def quote_string(var):
    ret = var
    if type(var) == str:
        ret = "\'%s\'" % (var)
    return ret

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def overwrite_params(params, ov):
    for i in ov:
        parts = i[0].split(".")
        access = "params"
        for k in parts:
            access += "[\"" + str(k) + "\"]"
        access += " = %s" % (str(quote_string(i[1])))
        exec(access)

def parse_commandline(argv):
    ovw = []
    if "-si" in argv:
        ovw.extend(set_int_param(argv))

    if "-sb" in argv:
        ovw.extend(set_bool_param(argv))

    if "-ss" in argv:
        ovw.extend(set_str_param(argv))    

    if "-ssl" in argv:
        ovw.extend(set_str_list_param(argv))

    if "-snl" in argv:
        ovw.extend(set_num_list_param(argv))

    return ovw

def set_int_param(argv):
    prev_si = 0
    ov = []
    for i in xrange(argv.count('-si')):
        ex_pos = argv.index('-si', prev_si+1)
        a = argv[ex_pos+1]
        prev_si = ex_pos
        a = a.split("=")
        ov.append(((a[0], int(a[1]))))

    return ov

def set_bool_param(argv):
    prev_sb = 0
    ov = []
    for i in xrange(argv.count('-sb')):
        ex_pos = argv.index('-sb', prev_sb + 1)
        a = argv[ex_pos + 1]
        prev_sb = ex_pos
        a = a.split("=")
        ov.append(((a[0], str2bool(a[1]))))

    return ov

def set_str_param(argv):
    prev_ss = 0
    ov = []
    for i in xrange(argv.count('-ss')):
        ex_pos = argv.index('-ss', prev_ss+1)
        a = argv[ex_pos+1]
        prev_ss = ex_pos
        a = a.split("=")
        ov.append(((a[0], str(a[1]))))

    return ov

def set_str_list_param(argv):
    prev_ssl = 0
    ov = []
    for i in xrange(argv.count('-ssl')):
        ex_pos = argv.index('-ssl', prev_ssl+1)
        a = argv[ex_pos+1]
        prev_ssl = ex_pos
        a = a.split("=")
        ov.append(((a[0], a[1].strip('[]').split(','))))

    return ov
        

def set_num_list_param(argv):
    prev_snl = 0
    ov = []
    for i in xrange(argv.count('-snl')):
        ex_pos = argv.index('-snl', prev_snl+1)
        a = argv[ex_pos+1]
        prev_snl = ex_pos
        a = a.split("=")
        ov.append(((a[0], eval(a[1]))))

    return ov


