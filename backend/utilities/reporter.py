import logging
import sys
import os
import datetime
import time
import functools

class Reporter:
    """
    A logger to display the optimization progress and/or others reports
    """
    
    _FORMATS = ['%(message)s',
                '[%(levelname)s]:     %(message)s',
                '(%(asctime)s)[%(levelname)s]:     %(message)s',
                '(%(asctime)s)[%(levelname)s]:     %(name)s: %(message)s',
                '(%(asctime)s)[%(levelname)s]:     %(name)s %(funcName)s: %(message)s']
    
    _NUM_FORMATS = len(_FORMATS)

    _DATE_FORMATS = ["%d/%m/%y, %H:%M:%S",
                     "%d %b %y, %H:%M:%S"]
    
    _NUM_DATE_FORMATS = len(_DATE_FORMATS)

    def __init__(self, name=__name__, on_console=True, log_file_name=None, log_file_folder=None, info_format=3, date_format=0):
        """
        Initializes the reporter
        
        Keyword Arguments:
            name {str} -- name of the reporter (default: {__name__})
            on_console {bool} -- if True, log reports are shown in the screen (default: {True})
            log_file_name {str} -- name of a log file; if None, no log file is written at all (default: {None})
            log_file_folder {str} -- folder path for the log file; if None, the cwd is set as folder (default: {None})
            info_format {int} -- index to select the info. format (default: {3})
                                 available formats:
                0                    ['%(message)s',
                1                   '[%(levelname)s]:     %(message)s',
                2                    '(%(asctime)s)[%(levelname)s]:     %(message)s',
                3                    '(%(asctime)s)[%(levelname)s]:     %(name)s: %(message)s',
                4                    '(%(asctime)s)[%(levelname)s]:     %(name)s %(funcName)s: %(message)s']

            date_format {int} -- index to select the date format; (default: {0})
                                 available formats:
                0                   ["%d/%m/%y, %H:%M:%S",
                1                    "%d %b %y, %H:%M:%S"]
        """
        self._logger = logging.getLogger(name=name)
        self._logger.setLevel(logging.INFO)

        self._on_console = bool(on_console)
        
        self._log_file_name = log_file_name
        self._log_file_folder = os.getcwd() if log_file_folder is None else log_file_folder

        self._info_format_id = info_format
        self._date_format_id = date_format
        self._check_formats()

        self._set_stream_handler()
        self._set_file_handler()

    def _check_formats(self):
        if not isinstance(self._info_format_id, int) or self._info_format_id not in list(range(0, Reporter._NUM_FORMATS)):
            raise ValueError('Invalid info format. It must be an integer in the range [0, {}]'.format(Reporter._NUM_FORMATS - 1))

        if not isinstance(self._date_format_id, int) or self._date_format_id not in list(range(0, Reporter._NUM_DATE_FORMATS)):
            raise ValueError('Invalid info format. It must be an integer in the range [0, {}]'.format(Reporter._NUM_DATE_FORMATS - 1))

    @property
    def formatter(self):        
        return logging.Formatter(fmt=Reporter._FORMATS[self._info_format_id],
                                 datefmt=Reporter._DATE_FORMATS[self._date_format_id])

    def _set_stream_handler(self):
        if self._on_console:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(self.formatter)
            self._logger.addHandler(handler)

    def _set_file_handler(self):
        if self._log_file_name is not None:
            try:
                file_path = os.path.join(str(self._log_file_folder), str(self._log_file_name) + '.log')
                handler = logging.FileHandler(file_path, mode='a')
            except:
                raise ValueError('Invalid folder for log file: {}'.format(self._log_file_folder))
            
            handler.setLevel(logging.INFO)
            handler.setFormatter(self.formatter)
            self._logger.addHandler(handler)

    def info(self, msg):
        self._logger.info(msg)

    def warn(self, msg):
        self._logger.warn(msg)
    
    def error(self, msg, terminate=True, exc_info=False):
        # if exc_info, it shows the exception information
        self._logger.error(msg, exc_info=exc_info)
        if terminate:
            sys.exit()


    def timer(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            t0 = datetime.datetime.now()
            res = func(*args, **kwargs)
            t1 = datetime.datetime.now()
            elapsed_time = t1 - t0
            self.info('Finished {} in {}'.format(func.__name__, elapsed_time))
            return res 
        
        return wrapped_func

    
    def runtime_debug(self, func):
        """Print the function signature and return value"""
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args_repr = [repr(a) for a in args]                      
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  
            signature = ", ".join(args_repr + kwargs_repr)           
            self.info(f"Calling {func.__name__}({signature})")
            value = func(*args, **kwargs)
            self.info(f"{func.__name__!r} returned {value!r}")       
            return value
        return wrapped_func

