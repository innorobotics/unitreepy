import sys
import traceback

def parseException(e):
    exception_type, exception_object, exception_traceback = sys.exc_info()
    filename = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    
    return ''.join(traceback.format_exception(type(e), e, e.__traceback__))
